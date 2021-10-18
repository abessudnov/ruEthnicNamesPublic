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

parser = ArgumentParser("Dump users from city, using VK API")
parser.add_argument("--folder",
                    help="result folder path. Default: data/data_by_city", default='data/data_by_city')
parser.add_argument("--city_id",
                    help="Use city by id https://vk.com/dev/database.getCities")
parser.add_argument("--city_name",
                    help="Find city by name https://vk.com/dev/database.getCities. country parameter has to be "
                         "provided")
parser.add_argument("--country",
                    help="country ISO code: https://vk.com/dev/country_codes. Default: RU", default='RU')
parser.add_argument("--algo",
                    help="fast - iterate over month, status, sex. slow - iterate over days, month, sex", default='fast')

args = parser.parse_args()

api_version = 5.52

access_token = os.getenv("access_token")

initial_big_delay = 60 * 20  # seconds to wait if reached restrictions
initial_small_delay = 2  # seconds to wait between each requests

logger = logging.getLogger('crawler_logger')

small_delay = initial_small_delay

whole_result = []

month_num_to_name = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

fields_possible_values = {
    "sex": [1, 2],
    "status": list(range(1, 9)),
    "birth_month": list(range(1, 13)),
    "days": list(range(1, 367))
}

ban_check_day = 1
ban_check_city = 1


def config_logger(result_folder):
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


def get_month_day_from_year_day(start_day_in_year):
    """ Converts day in the year to month and day of the month
    Args:
        :param start_day_in_year: day in the year (up to 366)
        :return: month, day of the month
    """
    days_passed_per_month = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    current_month = 1
    while days_passed_per_month[current_month] < start_day_in_year:
        current_month += 1
    current_day_in_month = start_day_in_year - days_passed_per_month[current_month - 1]
    return current_month, current_day_in_month


async def get_users(search_params, session):
    """ Sends requests to VK API, using search_params
    Args:
        :param search_params: params to send to users.search https://vk.com/dev/users.search
        :param session: aiohttp.ClientSession
    """
    logger.info(f"Starting search with params: {search_params}")

    async with session.get('https://api.vk.com/method/users.search', params=search_params) as response:
        logger.debug("users.search response")
        logger.debug((await response.text()))
        users = (await response.json())["response"]["items"]

    ids = []
    for user in users:
        ids.append(str(user["id"]))

    get_params = {
        "access_token": access_token,
        "user_ids": ','.join(ids),
        "fields": "sex,city,sex,bdate,personal",
        "name_case": "Nom",
        "v": str(api_version)
    }

    async with session.post('https://api.vk.com/method/users.get', data=get_params) as response:
        logger.debug("users.get response")
        logger.debug(await response.text())
        users = await response.json()
        return users


async def get_city_id(city_name, country_code, session):
    """ Using VK API finds city_id by it's name and country code
    Args:
        :param city_name: city name
        :param country_code: ISO country code, can be found here https://vk.com/dev/country_codes
        :param session: aiohttp.ClientSession
        :return: city_id
    """
    logger.info(f"Looking for city {city_name} in country {country_code}")

    getCountries_params = {
        "code": country_code,
        "access_token": access_token,
        "v": str(api_version)
    }
    async with session.get('https://api.vk.com/method/database.getCountries', params=getCountries_params) as response:
        resp = await response.json()
        country_id = resp["response"]["items"][0]["id"]

    search_params = {
        "country_id": country_id,
        "q": city_name,
        "access_token": access_token,
        "v": str(api_version)
    }
    async with session.get('https://api.vk.com/method/database.getCities', params=search_params) as response:
        resp = await response.json()
        logger.info(f"Found {resp['response']['count']} cities. Using {resp['response']['items'][0]}")
        return resp['response']['items'][0]['id']


async def check_ban(session):
    """ Check if we were banned by VK API, retrieving users from Moscow, and increasing day of birth on each call
    Args:
        :param session: aiohttp.ClientSession
        :return: bool showing ban status
    """
    birth_month, birth_day = get_month_day_from_year_day(ban_check_day)

    logger.info("Result was empty. Probably reached restrictions")
    logger.info(
        f"Sleeping for {small_delay} seconds and trying to get city {globals()['ban_check_city']} date {birth_day} {month_num_to_name[birth_month]} to check")

    await asyncio.sleep(small_delay)

    globals()['ban_check_day'] += 1
    if globals()['ban_check_day'] == 367:
        globals()['ban_check_city'] += 1
        globals()['ban_check_day'] = 1

    check_params = {
        "city": globals()['ban_check_city'],
        "birth_month": birth_month,
        "birth_day": birth_day,
        "sex": 1,
        "count": 1000,
        "v": str(api_version),
        "access_token": access_token,
    }
    check_result = await get_users(check_params, session=session)
    if len(check_result["response"]) != 0:
        logger.info(
            f"{birth_day} {month_num_to_name[birth_month]} returned {len(check_result['response'])}. No ban. Continue with main city")
        return False
    return True


def process_data(users, city_id):
    """ Converts users from response to needed format
    Args:
        :param users: users returned from VK API
        :param city_id: str
    """
    result = []
    for user in users:
        processed_user = {
            "id": user["id"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "city_id": city_id,
        }
        if "sex" in user:
            processed_user["sex"] = user["sex"]
        if "personal" in user and "langs" in user["personal"]:
            processed_user["langs"] = user["personal"]["langs"]
        if "bdate" in user:
            processed_user["bdate"] = user["bdate"]

        result.append(processed_user)

    return result


async def get_users_with_ban_check(params, session):
    """ Sends request to VK Api. In case of ban, waits for big_delay and retries request
    """
    result = await get_users(params, session=session)

    big_delay = initial_big_delay
    while len(result["response"]) == 0 and await check_ban(session):
        logger.info(f"Ban. Waiting for {big_delay} seconds and trying again")
        await asyncio.sleep(big_delay)
        result = await get_users(params, session=session)
        big_delay *= 2  # each time increasing delay

    logger.info(f"Number of users in result: {len(result['response'])}")
    return result


async def algo_iteration(params, result_file_path, session):
    """ Gets users from VK, converts to needed format, saves part to csv file and adds data to whole_result
    Args:
        :param params: params to send with VK API call
        :param result_file_path: path to file to save iteration results
        :param session: aiohttp.ClientSession
    """
    result = await get_users_with_ban_check(params, session)
    if not result or "response" not in result or len(result['response']) == 0:
        return
    data = process_data(result["response"], city_id=params["city"])
    logger.info(f"Saving current results to file {result_file_path}")
    pd.DataFrame(data).to_csv(result_file_path, index=False)

    globals()["whole_result"] += result["response"]


async def day_month_sex_algo(city_id, session):
    """ Iterate over users living in provided city, by sex, birth day and month.
    Args:
        :param city_id: can be found here https://vk.com/dev/database.getCities
        :param session: aiohttp.ClientSession session
    """
    number_of_iterations = 366 * 2
    logger.info(f"Total number of iterations: {number_of_iterations}")
    result_folder = os.path.join(dir, args.folder, f"{city_id}{args.algo}")

    for day in range(1, 366):
        for sex in range(1, 3):
            birth_month, birth_day = get_month_day_from_year_day(day)
            params = {
                "city": city_id,
                "birth_month": birth_month,
                "birth_day": birth_day,
                "sex": sex,
                "count": 1000,
                "v": str(api_version),
                "access_token": access_token
            }

            filename = f"sex_{sex}_day_{day}.csv"
            await algo_iteration(params, f"{result_folder}/{filename}", session)

            logger.info(f"Sleeping for {small_delay} seconds before next iteration")
            await asyncio.sleep(small_delay)


async def sex_status_month_algo(city_id, session):
    """ Iterate over users living in provided city, by sex, marriage status and birth month
    Args:
        :param city_id: can be found here https://vk.com/dev/database.getCities
        :param session: aiohttp.ClientSession session
    """
    number_of_iterations = 8 * 2 * 12
    logger.info(f"Total number of iterations: {number_of_iterations}")
    result_folder = os.path.join(dir, args.folder, f"{city_id}{args.algo}")

    for month in range(1, 13):
        for status in range(1, 9):
            for sex in range(1, 3):
                params = {
                    "city": city_id,
                    "birth_month": month,
                    "status": status,
                    "sex": sex,
                    "count": 1000,
                    "v": str(api_version),
                    "access_token": access_token
                }

                filename = f"sex_{sex}_status_{status}_month_{month}.csv"
                await algo_iteration(params, f"{result_folder}/{filename}", session)

                # need to wait, not to get banned by VK, because of requests flood
                logger.info(f"Sleeping for {small_delay} seconds before next request")
                await asyncio.sleep(small_delay)


async def get_whole_city(city_id='', city_name='', country=''):
    """ Downloads users living in city. Saves result to csv file
    Args:
        :param city_id: can be found here https://vk.com/dev/database.getCities
        :param city_name: if city_id was not provided, will be used to find id of the city by it's name and country ISO code
        :param country: if city_id was not provided, will be used to find id of the city by it's name and country ISO code
    """
    try:
        assert (city_id or (city_name and country)), "Either city_id or city_name with country has to be provided"

        async with aiohttp.ClientSession(headers={'Connection': 'keep-alive'}) as session:
            if not city_id:
                city_id = await get_city_id(city_name, country, session)

            result_folder = os.path.join(dir, args.folder, f"{city_id}{args.algo}")
            os.makedirs(result_folder, exist_ok=True)

            config_logger(result_folder)

            if args.algo == 'fast':
                # if --algo parameter equals "fast" iterate users by sex, marriage status, and birth month
                await sex_status_month_algo(city_id, session)
            else:
                # if --algo parameter equals "slow" iterate users by sex, day and month of birth
                await day_month_sex_algo(city_id, session)

            result_filename = f"cityid{city_id}{args.algo}{date.today().strftime('%d%m%Y')}.csv"
            logger.info(f"Saving whole result to file {result_folder}/{result_filename}")
            whole_data = process_data(whole_result, city_id)
            pd.DataFrame(whole_data).to_csv(f"{result_folder}/{result_filename}", index=False)

    except Exception as e:
        logger.error("Exception caught")
        logger.error(e, exc_info=True)


def main():
    asyncio.run(get_whole_city(args.city_id, args.city_name, args.country))


if __name__ == '__main__':
    main()
