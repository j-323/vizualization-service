import structlog

def get_logger(name: str):
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.JSONRenderer(),
        ]
    )
    return structlog.get_logger(name)