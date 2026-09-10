import io
import json

import httpx
import pytest

from scripts.evaluation.collect_http_metrics import sample


def test_timeout_preserves_last_sample_then_recovers():
    calls = 0

    def serve(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise httpx.ReadTimeout("delayed metrics", request=request)
        return httpx.Response(200 if calls < 4 else 500, text=f"metric {calls}\n")

    output, gaps = io.StringIO(), io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        assert sample(client, "http://engine/metrics", output, gaps)
        previous = output.getvalue()
        assert not sample(client, "http://engine/metrics", output, gaps)
        assert output.getvalue() == previous
        assert json.loads(gaps.getvalue())["error"] == "ReadTimeout"
        assert sample(client, "http://engine/metrics", output, gaps)
        assert output.getvalue().count("# sampled_at_s") == 2
        assert "metric 3" in output.getvalue()
        with pytest.raises(httpx.HTTPStatusError):
            sample(client, "http://engine/metrics", output, gaps)
