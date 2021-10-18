## VK data collection scripts

### Prerequisites

1. First you need to get a VK token. You can do this [here](https://oauth.vk.com/authorize?client_id=7634132&display=page&redirect_uri=https://oauth.vk.com/blank.html&scope=offline&response_type=token&v=5.124). After getting a token copy the parameter `access_token` (copy everything after `access_token=` and until `&`) and add it to the `.env` file in the current directory.

ATTENTION: Do not share your VK token.

2.  Install the required packages: `pip install -r requirements.txt`

### How to download data for all users in a city
    
Use `dump_city_delay.py`

You can get  `--city_id` from here:  https://vk.com/dev/database.getCities
 
Or you can use  `--city_name` and `--country` (https://vk.com/dev/country_codes)
 
An example for the city **Kursk**


`python dump_city_delay.py --city_id 75`

or 

`python dump_city_delay.py --city_name Курск --country RU`

the data will be saved to the folder `--folder`, by default `data/data_by_city`

To get help use `python dump_city_delay.py --help`

#### Choosing options for downloading

You can also specify the parameter  `--algo`

- `--algo fast` (default). The algorithm iterates by birth month, sex and marital status (12 * 2 * 8 = 192 iterations). It is faster but the data will be incomplete.

- `--algo slow`. The algorithm iterates by birth date, birth month and sex (366 * 2 = 732 iterations). It is slower, but the data are more complete. 

### How to download data from a VK community

Use `dump_groups.py` and pass to it `id` of communities.

You can specify the parameter `--folder` (by default: `data/data_from_groups`)

An example:

`python dump_groups.py 4684226 1777 17311245 --folder ./data/data_from_groups/tatars`

will collect the data from VK communities with ids `4684226 1777 17311245` and save to the folder `./data/data_from_groups/tatars` (with a separate file for each community).

To get help use `python dump_groups.py --help`