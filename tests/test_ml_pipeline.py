import unittest

import pandas as pd

from features import FEATURE_COLUMNS, build_feature_dataset, build_prediction_features
from modeling import train_model
from predictor import initialize_predictor, predict_match


class MLPipelineTests(unittest.TestCase):
    def test_training_and_prediction_with_local_matches(self):
        matches = pd.DataFrame([
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "home_team": "Team A",
                "away_team": "Team B",
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
            for index, (home_goals, away_goals) in enumerate([(2, 0), (1, 1), (0, 2)] * 10)
        ])
        features = build_feature_dataset(matches)
        self.assertEqual(len(features), 2 * len(matches))
        self.assertEqual(features.iloc[0]["team_a_form_points_last5"], 0)
        self.assertEqual(features.iloc[0]["team_b_form_points_last5"], 0)
        training = train_model(features)
        vector = build_prediction_features("Team A", "Team B", matches)
        self.assertEqual(list(vector.columns), FEATURE_COLUMNS)
        probabilities = training["model"].predict_proba(vector)[0]
        self.assertAlmostEqual(float(sum(probabilities)), 1.0)
        initialize_predictor(training["model"], matches)
        result = predict_match("Team A", "Team B")
        self.assertIn("Vorhersage:", result)
        self.assertIn("Wahrscheinlichkeit", result)
        self.assertIn("Keine Spielerstatistiken verfügbar", result)
