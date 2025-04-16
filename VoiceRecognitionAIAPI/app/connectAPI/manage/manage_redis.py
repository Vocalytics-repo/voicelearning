import redis
# Redis 연결
# manage_redis.py
def connect_redis(minutes = 30, host = 'localhost', port=6379, db=0):
    """
        Redis 클라이언트를 생성하고 설정된 만료 시간을 반환합니다.

        Args:
            minutes (int): 캐시 데이터 만료 시간(분)
            host (str): Redis 서버 호스트
            port (int): Redis 서버 포트
            db (int): Redis DB 번호

        Returns:
            tuple: (redis_client, expiry_time(초))
        """

    redis_client = redis.Redis(host=host, port=port, db=db)
    REDIS_EXPIRY = 60 * minutes  # 30분 캐시 유지

    return redis_client, REDIS_EXPIRY

