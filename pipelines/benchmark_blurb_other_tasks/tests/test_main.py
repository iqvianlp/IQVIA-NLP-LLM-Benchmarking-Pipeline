from pipelines.benchmark_blurb_other_tasks.main import Runner


class TestParsePrediction:

    runner = Runner(None, None, 'dummy_output_dir', None, None, None, None)

    def test_hoc_with_valid_responses_only_check_only_valid_codes(self):
        is_valid, prediction = self.runner._parse_prediction(
            '"evading growth suppressors","tumor promoting inflammation"',
            'HoC',
            None
        )
        assert is_valid
        assert set(prediction) == {'GS', 'TPI'}

    def test_hoc_with_mixed_responses_check_only_valid_codes(self):
        is_valid, prediction = self.runner._parse_prediction(
            '"evading growth suppressors","tumor promoting inflammation","I love ice-cream!"',
            'HoC',
            None
        )
        assert is_valid
        assert set(prediction) == {'GS', 'TPI'}

    def test_hoc_with_invalid_response_check_empty_codes_and_output_marked_as_invalid(self):
        is_valid, prediction = self.runner._parse_prediction(
            '',
            'HoC',
            None
        )
        assert not is_valid
        assert set(prediction) == set()
