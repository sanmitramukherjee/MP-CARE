"""802.11 DCF extension tests.

Invariants:
1. With dcf_enabled=False the RNG stream and delivery decisions are identical
   to the historical Bernoulli path (opt-in model, no regression).
2. With DCF enabled, a broadcast is one frame, not one per neighbour.
3. Hidden terminals collide at a common receiver; carrier-sensed neighbours
   defer instead.
4. Synthetic background load consumes airtime in proportion to its rate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "network_model"))

from network_model.model import LinkModel, NetworkParams  # noqa: E402
from network_model.dcf import DcfChannel, DcfParams  # noqa: E402


def test_dcf_off_preserves_legacy_draws():
    a = LinkModel(NetworkParams(rng_seed=11))
    b = LinkModel(NetworkParams(rng_seed=11))
    for d in (40.0, 120.0, 240.0, 299.0):
        for (x, y) in ((0, 1), (0, 2), (1, 2)):
            assert a.unicast_success(d, x, y) == b.unicast_success(d, x, y)
    assert a.dcf is None


def test_dcf_high_prr_delivers_every_packet():
    p = NetworkParams(rng_seed=5, dcf_enabled=True)
    lm = LinkModel(p)
    pos = {0: (0.0, 0.0, 40.0), 1: (80.0, 0.0, 40.0)}
    for i in range(20):
        lm.begin_tick(i * 0.1, pos)
        assert lm.unicast_success(80.0, 0, 1)
    assert lm.dcf.packets_ok == 20


def test_broadcast_schedules_one_frame():
    p = NetworkParams(rng_seed=3, dcf_enabled=True)
    lm = LinkModel(p)
    pos = {0: (0.0, 0.0, 40.0), 1: (100.0, 0.0, 40.0),
           2: (0.0, 100.0, 40.0), 3: (100.0, 100.0, 40.0)}
    lm.begin_tick(0.0, pos)
    out = lm.broadcast_success({1: 1.0, 2: 1.0, 3: 1.0}, 0, size_bytes=64)
    assert len(out) == 3
    assert lm.dcf.frames_scheduled == 1


def test_hidden_terminal_collision():
    p = NetworkParams(rng_seed=2, dcf_enabled=True, dcf_cs_range_m=500.0)
    lm = LinkModel(p)
    # A and C are 520 m apart (outside the 500 m carrier-sense range) but
    # both are within decode range of R: classic hidden terminal.
    pos = {0: (0.0, 0.0, 40.0), 1: (250.0, 0.0, 40.0),
           2: (520.0, 0.0, 40.0)}
    lm.begin_tick(0.0, pos)
    assert lm.unicast_success(250.0, 0, 1, size_bytes=1000)
    # C cannot hear A and therefore does not defer; its frame overlaps A's at R
    lm.unicast_success(270.0, 2, 1, size_bytes=1000)
    assert lm.dcf.collisions >= 1


def test_broadcast_collision_counters():
    """A hidden transmitter blocking a broadcast is counted per receiver."""
    class _FixedRng:
        def randrange(self, n):
            return 0

        def random(self):
            return 0.0

    dcf = DcfChannel(_FixedRng(), DcfParams(cs_range_m=500.0))
    pos = {0: (0.0, 0.0, 40.0), 1: (250.0, 0.0, 40.0),
           2: (520.0, 0.0, 40.0)}
    dcf.begin_tick(0.0, pos)
    assert dcf.attempt_broadcast(0, {1: 1.0}) == {1: True}
    # node 2 is 520 m from node 0 (outside carrier sense) but its frame
    # overlaps at receiver 1, which can hear both transmitters
    assert dcf.attempt_broadcast(2, {1: 1.0}) == {1: False}
    assert dcf.broadcast_frames == 2
    assert dcf.broadcast_rx_attempts == 2
    assert dcf.broadcast_rx_blocked == 1
    assert dcf.broadcast_rx_ok == 1
    assert dcf.broadcast_collided_frames == 1
    s = dcf.summary()
    assert s["broadcast_rx_blocked"] == 1
    assert s["broadcast_collided_frames"] == 1


def test_background_load_airtime():
    p = NetworkParams(rng_seed=1, dcf_enabled=True,
                      dcf_bg_frames_per_s=600.0)
    lm = LinkModel(p)
    lm.begin_tick(0.0, {i: (i * 50.0, 0.0, 40.0) for i in range(5)})
    assert lm.dcf.bg_frames == 60
    assert 0.04 < lm.dcf.bg_airtime_s < 0.07


def test_dcf_reproducible():
    def once():
        p = NetworkParams(rng_seed=9, dcf_enabled=True,
                          shadow_sigma_db=2.0)
        lm = LinkModel(p)
        pos = {i: (i * 120.0, 0.0, 40.0) for i in range(4)}
        outs = []
        for tick in range(30):
            lm.begin_tick(tick * 0.1, pos)
            lm.step_shadow([(0, 1), (0, 2), (1, 2), (2, 3)], 0.1)
            for (a, b) in ((0, 1), (1, 2), (2, 3)):
                outs.append(lm.unicast_success(120.0, a, b, size_bytes=1000))
        return outs, lm.dcf.summary()
    x, y = once(), once()
    assert x == y
