import threading

from hotelmon.logdb import PredictionLog


def _rec(i: int) -> dict:
    return {"arrival_date": "2017-01-0" + str(1 + i % 9), "lead_time": i, "hotel": "City Hotel"}


def test_roundtrip_and_label_join(tmp_path):
    log = PredictionLog(tmp_path / "x.db")
    ids = log.log_predictions("req1", "v1", [_rec(1), _rec(2)], [0.1, 0.9])
    assert ids == [1, 2]
    stored, unknown = log.log_labels([(1, 0), (5, 1)])
    assert (stored, unknown) == (1, [5])
    df = log.read()
    assert list(df["prediction_id"]) == [1, 2]
    assert list(df["lead_time"]) == [1, 2]
    assert df.loc[0, "label"] == 0
    assert df["label"].isna().iloc[1]
    assert list(df["probability"]) == [0.1, 0.9]
    log.close()


def test_label_overwrite_keeps_latest(tmp_path):
    log = PredictionLog(tmp_path / "x.db")
    log.log_predictions("r", "v1", [_rec(1)], [0.5])
    log.log_labels([(1, 0)])
    log.log_labels([(1, 1)])
    assert log.read()["label"].tolist() == [1]
    log.close()


def test_persists_across_connections(tmp_path):
    p = tmp_path / "x.db"
    log = PredictionLog(p)
    log.log_predictions("r", "v1", [_rec(1)], [0.5])
    log.close()
    assert PredictionLog(p).count() == 1


def test_concurrent_writes(tmp_path):
    log = PredictionLog(tmp_path / "x.db")

    def worker(k):
        for i in range(25):
            log.log_predictions(f"r{k}-{i}", "v1", [_rec(i)], [0.5])

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert log.count() == 100
    assert log.read()["prediction_id"].is_unique
    log.close()


def test_empty_log_reads_empty_frame(tmp_path):
    assert PredictionLog(tmp_path / "x.db").read().empty
