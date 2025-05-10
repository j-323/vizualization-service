import redis

class RedisCache:
    def __init__(self, url: str):
        self.cli = redis.from_url(url)

    def get(self, key: str):
        return self.cli.get(key)

    def set(self, key: str, val: str, ex: int = 300):
        self.cli.set(key, val, ex=ex)