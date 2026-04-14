import unittest
from unittest.mock import patch, MagicMock
import time
import sys

# Mock agent module to prevent loading torch and heavy models if not necessary
mock_agent = MagicMock()
sys.modules['agent'] = mock_agent

from risk_manager import RiskManager
from config import (
    LEVERAGE, FEE_OPEN_LEVERAGED, FEE_CLOSE_LEVERAGED,
    LOCK_SECONDS, STOP_LOSS_PCT, TRAIL_ACTIVATE, TRAIL_DRAWDOWN,
    DEAD_THRESHOLD
)

class MockArena:
    def calculate_reward(self, entry, price_now, pos, smc_info):
        return 0.1

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.num_agents = 2
        self.rm = RiskManager(self.num_agents)
        self.arena = MockArena()

    @patch('time.time')
    def test_open_position(self, mock_time):
        mock_time.return_value = 1000.0

        # Agent 0 goes LONG
        action = 1
        price_now = 100.0

        brain_score, cur_pnl, total_pnl, actual_action = self.rm.update_pnl(0, action, price_now, self.arena)

        self.assertEqual(actual_action, 1)
        self.assertEqual(self.rm.stats["0"]["pos"], 1)
        self.assertEqual(self.rm.stats["0"]["entry"], 100.0)

        # cur_pnl should be -FEE_OPEN_LEVERAGED on open
        expected_cur_pnl = round(-FEE_OPEN_LEVERAGED, 4)
        self.assertAlmostEqual(cur_pnl, expected_cur_pnl)
        self.assertAlmostEqual(total_pnl, expected_cur_pnl)

    @patch('time.time')
    def test_stop_loss(self, mock_time):
        mock_time.return_value = 1000.0

        # Open position
        self.rm.update_pnl(0, 1, 100.0, self.arena)

        # Fast forward past LOCK_SECONDS
        mock_time.return_value = 1000.0 + LOCK_SECONDS + 1

        # Calculate a price that triggers stop loss
        # STOP_LOSS_PCT is -8.0
        # pure_money = pnl_pct * LEVERAGE * 100 - FEE_OPEN_LEVERAGED
        # pnl_pct = (price - 100) / 100
        # We need pure_money <= -8.0
        # pnl_pct * 3000 - 1.5 <= -8.0
        # pnl_pct * 3000 <= -6.5
        # pnl_pct <= -0.002166...
        # So price = 100 * (1 - 0.003) = 99.7 should be enough
        price_now = 99.7
        # Verify it drops enough
        # pure_money = (-0.003 * 3000) - 1.5 = -9.0 - 1.5 = -10.5 <= -8.0

        # Try to hold (action=1)
        brain_score, cur_pnl, total_pnl, actual_action = self.rm.update_pnl(0, 1, price_now, self.arena)

        # Action should be forced to 0 (stop loss)
        self.assertEqual(actual_action, 0)
        self.assertEqual(self.rm.stats["0"]["pos"], 0)

    @patch('time.time')
    def test_trailing_stop(self, mock_time):
        mock_time.return_value = 1000.0
        self.rm.update_pnl(0, 1, 100.0, self.arena)
        mock_time.return_value = 1000.0 + LOCK_SECONDS + 1

        # Hit TRAIL_ACTIVATE
        # pure_money >= 2.0
        # pnl_pct * 3000 - 1.5 >= 2.0
        # pnl_pct * 3000 >= 3.5
        # pnl_pct >= 0.001166...
        price_high = 100.2
        # pure_money = (0.002 * 3000) - 1.5 = 6.0 - 1.5 = 4.5

        self.rm.update_pnl(0, 1, price_high, self.arena)
        self.assertTrue(self.rm.stats["0"]["trail_active"])
        self.assertAlmostEqual(self.rm.stats["0"]["peak_pnl"], 4.5)

        # Drop by TRAIL_DRAWDOWN (1.5)
        # We need pure_money <= 4.5 - 1.5 = 3.0
        # pnl_pct * 3000 - 1.5 = 2.5
        # pnl_pct * 3000 = 4.0
        # pnl_pct = 0.00133...
        price_drop = 100.133
        # pure_money ~ 3.99 - 1.5 = 2.49 <= 3.0

        brain_score, cur_pnl, total_pnl, actual_action = self.rm.update_pnl(0, 1, price_drop, self.arena)

        # Should trigger trailing stop and force close
        self.assertEqual(actual_action, 0)
        self.assertEqual(self.rm.stats["0"]["pos"], 0)

    @patch('time.time')
    def test_normal_close(self, mock_time):
        mock_time.return_value = 1000.0
        self.rm.update_pnl(0, 1, 100.0, self.arena)

        mock_time.return_value = 1000.0 + LOCK_SECONDS + 1

        price_now = 100.1 # Small profit
        # pure_money = 0.001 * 3000 - 1.5 = 3.0 - 1.5 = 1.5
        # realize = 1.5 - FEE_CLOSE_LEVERAGED = 1.5 - 1.5 = 0.0

        brain_score, cur_pnl, total_pnl, actual_action = self.rm.update_pnl(0, 0, price_now, self.arena)

        self.assertEqual(actual_action, 0)
        self.assertEqual(self.rm.stats["0"]["pos"], 0)
        # Assuming LEVERAGE=30, FEE_OPEN=1.5, FEE_CLOSE=1.5
        # Initial realized is 0
        self.assertAlmostEqual(self.rm.stats["0"]["realized_pnl"], 0.0)

    @patch('time.time')
    def test_holding_lock(self, mock_time):
        mock_time.return_value = 1000.0
        self.rm.update_pnl(0, 1, 100.0, self.arena)

        # Try to close before lock expires
        mock_time.return_value = 1000.0 + LOCK_SECONDS - 10

        brain_score, cur_pnl, total_pnl, actual_action = self.rm.update_pnl(0, 0, 100.1, self.arena)

        # Action should be ignored and original position maintained
        self.assertEqual(actual_action, 1)
        self.assertEqual(self.rm.stats["0"]["pos"], 1)

    @patch('time.time')
    def test_check_evolution_death(self, mock_time):
        mock_time.return_value = 1000.0

        self.rm.stats["0"]["realized_pnl"] = DEAD_THRESHOLD - 1.0

        mock_agent_instance = MagicMock()
        mock_agent_instance.state_dim = 10
        mock_agent_instance.stack_size = 4
        mock_agent_instance.action_dim = 3
        mock_agent_instance.use_lstm = False

        new_agent = self.rm.check_evolution(0, mock_agent_instance, DEAD_THRESHOLD - 1.0)

        # Stats should be reset
        self.assertEqual(self.rm.stats["0"]["realized_pnl"], 0.0)
        self.assertEqual(self.rm.stats["0"]["pos"], 0)

        # A new agent is returned
        self.assertIsNotNone(new_agent)

if __name__ == '__main__':
    unittest.main()
