from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from loguru import logger
from pydantic import SecretStr

from backend.src.domain.enums import SerpType

if TYPE_CHECKING:
    pass


class SerpService:
    pass

    # initial active: Exa-ai, Tavily
    @staticmethod
    async def web_search(serper: SerpType):
        # TODO: switch to config - hardcode defaults into pref
        match serper:
            case SerpType.EXA:
                # NOTE: EXA AI
                from langchain_exa import ExaSearchResults

                # initialize the exasearchresults tool
                search_tool = ExaSearchResults(
                    exa_api_key=SecretStr("06acf324-3111-46b4-96b2-ded0e5ad88be")
                )

                # perform a search query
                search_results = search_tool._run(
                    query="weather in chongqing china",
                    num_results=1,
                    text_contents_options=None,
                    highlights=True,
                    summary=True,
                    type="neural",
                )

                from pprint import pp

                pp(search_results)

            # NOTE: EXA SHAPE

            # SearchResponse(results=[Result(url='https://weather.metoffice.gov.uk/forecast/wm5xzjb',
            #                                id='https://weather.metoffice.gov.uk/forecast/wm5xzjb',
            #                                title='Chongqing (China) weather - Met Office',
            #                                score=0.4930480059918553,
            #                                published_date=None,
            #                                author='',
            #                                image='https://weather.metoffice.gov.uk/forecast/static/images/common/icons/social_card.jpg',
            #                                favicon='https://weather.metoffice.gov.uk/favicon.png',
            #                                subpages=None,
            #                                extras=None,
            #                                text=None,
            #                                highlights=['# Chongqing (China) weather\n'
            #                                            'Feels like\n'
            #                                            'Daily high 21°C Maximum feels like '
            #                                            'temperature: 21° Celsius;\n'
            #                                            'Daily low\n'
            #                                            '16°C Minimum feels like '
            #                                            'temperature: 16° Celsius; \n'
            #                                            '## Feels like temperature ... Air '
            #                                            'pollution\n'
            #                                            'No air pollution data\n'
            #                                            '## Air pollution This shows the '
            #                                            'average air pollution levels for '
            #                                            'regions of the country. This can '
            #                                            'be from pollutants such as sulphur '
            #                                            'dioxide, nitrogen oxides, and '
            #                                            'particulate matter. The data is '
            #                                            "taken from Defra’s 'Daily Air "
            #                                            "Quality Index' . There is"],
            #                                highlight_scores=[0.07666015625],
            #                                summary='The current weather in Chongqing, '
            #                                        'China, is 21°C with sunny intervals '
            #                                        'and a 40% chance of rain. Over the '
            #                                        'next week, temperatures will range '
            #                                        'from a high of 23°C on Friday to a low '
            #                                        'of 17°C on Sunday, with varying '
            #                                        'conditions including sunny intervals, '
            #                                        'overcast skies, and light rain '
            #                                        'expected. Notably, rain is forecasted '
            #                                        'for Monday (40% chance) and Sunday '
            #                                        '(60% chance). For detailed hourly '
            #                                        'forecasts and updates, visit the [Met '
            #                                        'Office '
            #                                        'website](https://weather.metoffice.gov.uk/forecast/wm5xzjb).')],
            #                autoprompt_string=None,
            #                resolved_search_type='neural',
            #                auto_date=None,
            #                context=None,
            #                statuses=None,
            #                cost_dollars=CostDollars(total=0.007,
            #                                         search={'neural': 0.005},
            #                                         contents={'highlights': 0.001,
            #                                                   'summary': 0.001}))

            case SerpType.TAVILY:
                # NOTE: TAVILY

                from langchain_community.retrievers import TavilySearchAPIRetriever

                retriever = TavilySearchAPIRetriever(
                    k=3, api_key="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"
                )

                query = "weather in chongqing china"

                from pprint import pp

                pp(retriever.invoke(query))

                # NOTE: TAVILY SHAPE

                # [Document(metadata={'title': 'Zelda Breath of the Wild was released in 2017 (8 years ago ... - Reddit',
                # 'source': 'https://www.reddit.com/r/Breath_of_the_Wild/comments/1mgi6dv/zelda_breath_of_the_wild_was_released_in_2017_8/',
                # 'score': 0.9991374,
                # 'images': []},
                # page_content="Zelda Breath of the Wild was released in 2017 (8 years ago). I picked it up again to play on my Nintendo Switch 2 in 2025 and I'm amazed"),

                #  Document(metadata={'title': 'The Legend of Zelda: Breath of the Wild',
                #  'source': 'https://www.zeldadungeon.net/wiki/The_Legend_of_Zelda:_Breath_of_the_Wild',
                #  'score': 0.99528164,
                #  'images': []},
                #  page_content='It was released simultaneously on the Wii U and Nintendo Switch on March 3, 2017. By March 31, 2023, worldwide sales exceeded 31.5 million units; 29.81 million')]

                pass
            case _:
                pass
