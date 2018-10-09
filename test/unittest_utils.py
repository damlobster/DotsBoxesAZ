import asyncio


def async_test(coro):
    def wrapper(self):
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro(self, loop))
    return wrapper
