import asyncio
import os
import aiohttp
import logging
import pandas as pd

from dotenv import load_dotenv
from argparse import ArgumentParser
from datetime import date

load_dotenv()

dir = os.path.dirname(__file__)

parser = ArgumentParser("Dump whole group members")
parser.add_argument('group_ids', type=str, nargs='+',
                    help='ids (single or multiple) of groups to dump')
parser.add_argument("--folder",
                    help="result folder path. Default: data/data_from_groups", default='data/data_from_groups')

args = parser.parse_args()

result_folder = os.path.join(dir, args.folder)
os.makedirs(result_folder, exist_ok=True)

api_version = 5.52

offset_per_request = 10000

logger = logging.getLogger('groups_crawler_logger')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(f'{result_folder}/{logger.name}-debug.log', mode='w')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

access_token = os.getenv("access_token")


async def get_group_users_slice(group_id, total_count, session,
                                offset=0):
    """ Gets up to 20000 users of VK group, calling VK API "execute" method. Code of getGroupSlice method can be found in ./VKexecuteGetGroupSlice.js
    :param group_id: id of VK group
    :param total_count: total number of group members
    :param session: aiohttp.ClientSession
    :param offset: from which user to start
    """
    params = {
        "access_token": access_token,
        "group_id": group_id,
        "total_count": total_count,
        "offset": offset,
        "v": str(api_version)
    }

    async with session.get('https://api.vk.com/method/execute.getGroupSlice', params=params) as response:
        resp = await response.json()
        return resp


async def start_with_delay(delay, foo, *args, **kwargs):
    """
    Execute foo after delay
    """
    await asyncio.sleep(delay)
    return await foo(*args, **kwargs)


async def get_whole_group(group_id):
    """ Downloads members of VK Group
    Args:
        :param group_id: id of VK Group
    """
    try:
        async with aiohttp.ClientSession(headers={'Connection': 'keep-alive'}) as session:
            async with session.get('https://api.vk.com/method/groups.getById',
                                   params={
                                       "group_id": group_id,
                                       "fields": "members_count",
                                       "access_token": access_token,
                                       "v": str(api_version)
                                   }) as response:
                group_size = (await response.json())["response"][0]["members_count"]
            logger.info("Group size : " + str(group_size))
            result = []
            number_of_iterations = group_size // offset_per_request

            logger.info(f"Total number of iterations: {number_of_iterations + 1}")
            tasks = []
            for i in range(number_of_iterations + 1):
                logger.info(f"Starting iteration {i + 1} with delay: {i * 0.4}")
                tasks.append(
                    start_with_delay(i * 0.4, get_group_users_slice, group_id, group_size,
                                     session=session, offset=offset_per_request * i)
                )

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for response in responses:
                if type(response) is dict:
                    result += process_data(response["response"])
                else:
                    logger.debug(response)
            result_filename = f"{result_folder}/group{group_id}_{date.today().strftime('%d%m%Y')}.csv"
            logger.info(f"Writing to the file {result_filename}")
            pd.DataFrame(result).to_csv(result_filename, index=False)

    except Exception as e:
        logger.error(e, exc_info=True)


def process_data(users):
    """ Converts users from response to needed format
    Args:
        :param users: users returned from VK API
    """
    result = []
    for user in users:
        processed_user = {
            "id": user["id"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
        }
        if "city" in user:
            processed_user["city_id"] = user["city"]["id"]
        if "sex" in user:
            processed_user["sex"] = user["sex"]
        if "personal" in user and "langs" in user["personal"]:
            processed_user["langs"] = user["personal"]["langs"]
        if "bdate" in user:
            processed_user["bdate"] = user["bdate"]

        result.append(processed_user)

    return result


async def get_groups(group_ids):
    for group_id in group_ids:
        await get_whole_group(group_id)


def main():
    asyncio.run(get_groups(args.group_ids))


if __name__ == '__main__':
    main()
