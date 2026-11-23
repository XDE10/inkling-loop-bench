import re, time

class MockBackend:
    name="mock"
    def generate(self, question, config):
        start=time.perf_counter()
        # Deterministic test response; it is deliberately not a model prediction.
        answer=question["answer"]
        text=f"TEST DATA ONLY. The computed answer is {answer:g}."
        elapsed=(time.perf_counter()-start)*1000
        return {"text":text,"finish_reason":"stop","reasoning_tokens":0,"answer_tokens":8,
                "ttft_ms":elapsed,"tpot_ms":0.0,"e2e_ms":elapsed,"peak_vram_mb":None,
                "is_mock":True}
