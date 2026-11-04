import unittest
from pathlib import Path
from state_guidance.health_gate import wait_healthy, protection_due


class Clock:
    def __init__(self): self.now=100
    def time(self): return self.now
    def monotonic(self): return self.now
    def sleep(self,n): self.now+=n


class HealthTests(unittest.TestCase):
    def run_gate(self, observations, matches=lambda _:True):
        clock=Clock(); owner={'pid':1,'start_ticks':2}; events=[]
        def read(_):
            item=observations.pop(0) if len(observations)>1 else observations[0]
            if isinstance(item,BaseException):raise item
            return {'status':'watching','updated_unix':clock.time(),'coordinator':owner,'identity':{'pid':3},**item}
        wait_healthy(Path('/unused'),owner,grace_seconds=6,clock=clock,reader=read,matches=matches,
                     emit=lambda _,kind,**detail:events.append((kind,detail)))
        return clock,events
    def test_healthy_no_wait(self):self.assertEqual(self.run_gate([{}])[0].now,100)
    def test_read_error_recovers_without_accepting_error(self):
        clock,events=self.run_gate([OSError('transient'),{'status':'read_error'},{}])
        self.assertEqual(clock.now,104);self.assertEqual(events[-1][0],'coordinator_health_recovered')
    def test_stale_snapshot_recovers(self):self.assertEqual(self.run_gate([{'updated_unix':1},{}])[0].now,102)
    def test_persistent_stale_fails(self):
        with self.assertRaisesRegex(RuntimeError,'age_seconds'):self.run_gate([{'updated_unix':1}])
    def test_wrong_owner_not_accepted(self):
        with self.assertRaises(RuntimeError):self.run_gate([{'coordinator':{'pid':1,'start_ticks':9}}])
    def test_dead_watchdog_not_accepted(self):
        with self.assertRaises(RuntimeError):self.run_gate([{}],matches=lambda _:False)
    def test_user_interrupt_not_swallowed(self):
        with self.assertRaises(KeyboardInterrupt):self.run_gate([KeyboardInterrupt()])
    def test_watchdog_persistent_failure_boundary(self):
        self.assertFalse(protection_due(2,0,61));self.assertFalse(protection_due(3,0,59.9))
        self.assertTrue(protection_due(3,0,60))


if __name__=='__main__':unittest.main()
