from telethon import TelegramClient
import telethon_conf as conf

api_id = conf.api_id
api_hash = conf.api_hash
client = TelegramClient('me', api_id, api_hash)


async def make(message=None, image=None):
    if message is not None:
        await client.send_message('me', message)
    if image is not None:
        await client.send_file('me', image)


def alarm(message=None, image=None):
    with client:
        client.loop.run_until_complete(make(message, image))


if __name__ == '__main__':
    alarm('zidane', 'zidane.jpg')
    alarm('zidane')
    alarm(None, 'zidane.jpg')
