"""The certificate lane keeps an active verifier and replaces it after a timeout."""
import json
from unittest import TestCase
from unittest.mock import Mock, patch

from engine import general_resolution as graphs
from engine.polynomial import parse


class CertificateWorkerSpare(TestCase):
    def test_rejection_reuses_first_worker_and_timeout_enables_replacement(self):
        names = list("xyz")
        poly = parse("x^2-y^2*z", names)
        schedules = []
        for early_spare in (False, True):
            with self.subTest(early_spare=early_spare):
                first, replacement = Mock(answered=0), Mock(answered=0)
                first.ask.side_effect = [json.dumps(["rejected", None, None]), None]
                replacement.ask.return_value = json.dumps(["certified", {"certificate": {}}, None])
                workers = Mock()
                spares = []

                def checkout(*, spare=True):
                    spares.append(spare)
                    return first if len(spares) <= 2 else replacement

                workers.take.side_effect = checkout
                with patch.object(graphs, "_workers", workers):
                    record = graphs.search(poly, names, "embedded", exhaustive=True, prewarm_spare=early_spare)
                self.assertEqual(record["certificate"]["candidates_examined"], 3)
                self.assertEqual(spares, [early_spare, early_spare, True])
                workers.warm.assert_called_once_with()
                first.close.assert_called_once_with()
                replacement.close.assert_not_called()
                calls = [*first.ask.call_args_list, *replacement.ask.call_args_list]
                schedules.append([json.loads(call.args[0]) for call in calls])
        self.assertEqual(schedules[0], schedules[1])
