import core.metrics as m
print("RUNTIME METRICS:", m.__file__, "has total_delay_ms?", hasattr(m, "total_delay_ms"))
from ui.app import run_app

if __name__ == "__main__":
    run_app()

