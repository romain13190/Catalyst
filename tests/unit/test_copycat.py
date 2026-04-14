class TestDetectIndexCopycats:
    def test_no_overlap(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 2, 3}, "upload_time": 100.0},
            "miner_b": {"indices": {4, 5, 6}, "upload_time": 101.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {}

    def test_overlap_later_uploader_loses(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 2, 5}, "upload_time": 100.0},
            "miner_b": {"indices": {3, 4, 5}, "upload_time": 200.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {"miner_b": {5}}

    def test_overlap_earlier_uploader_keeps(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 5}, "upload_time": 300.0},
            "miner_b": {"indices": {5, 6}, "upload_time": 100.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {"miner_a": {5}}

    def test_three_miners_same_index(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {10}, "upload_time": 300.0},
            "miner_b": {"indices": {10}, "upload_time": 100.0},
            "miner_c": {"indices": {10}, "upload_time": 200.0},
        }
        rejected = detect_index_copycats(submissions)
        assert 10 in rejected.get("miner_a", set())
        assert 10 in rejected.get("miner_c", set())
        assert "miner_b" not in rejected or 10 not in rejected["miner_b"]

    def test_equal_timestamps_both_keep(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1}, "upload_time": 100.0},
            "miner_b": {"indices": {1}, "upload_time": 100.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {}

    def test_none_timestamp_not_rejected(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1}, "upload_time": None},
            "miner_b": {"indices": {1}, "upload_time": 100.0},
        }
        rejected = detect_index_copycats(submissions)
        assert rejected == {}

    def test_empty_submissions(self):
        from grail.validator.copycat import detect_index_copycats

        assert detect_index_copycats({}) == {}

    def test_single_miner(self):
        from grail.validator.copycat import detect_index_copycats

        submissions = {
            "miner_a": {"indices": {1, 2, 3}, "upload_time": 100.0},
        }
        assert detect_index_copycats(submissions) == {}
