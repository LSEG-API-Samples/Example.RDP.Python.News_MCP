from typing import Any
import json
import logging
from mcp.server.fastmcp import FastMCP
from rdp_auth import make_authenticated_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting news server")

mcp = FastMCP("news")

RDP_BASE_URL = "https://api.refinitiv.com"


@mcp.tool()
async def get_headlines(user_query: str) -> str:
    """
    Search for news articles using Refinitiv's advanced query syntax. Returns a simplified list of matching stories.

    This tool searches news headlines using sophisticated query language that supports boolean operators,
    company codes, geographic filters, date ranges, language selection, and news sources. Use this when you
    need to find news articles about specific topics, companies, people, or events with precise filtering.

    Args:
        user_query (str): Query string using News search syntax. Can be simple keywords or complex boolean expressions.

    Returns:
        str: JSON array of simplified story objects containing:
             - story_id: Unique identifier for the story (use this with get_news_story to get full details)
             - headline: The headline/title of the news article

    Example usage:
        Explicit FreeText (use quotes): Obtains headlines for stories having the text "electric car" or "electric vehicle" in their title.
        user_query = '"electric car" OR "electric vehicle"'

    SearchIn token (HeadlineOnly): Obtains headlines for stories having the text "Reports" or "Announces" and searchIn:HeadlineOnly in their title.
        user_query = '"Reports" or "Announces" and searchIn:HeadlineOnly'

    SearchIn token (FullStory): Obtains headlines for stories having the text "inflation" and searchIn:FullStory in their title or body.
        user_query = '"inflation" and searchIn:FullStory'

    English language (L:EN): Obtains headlines for stories being written in English, using the language code "L:EN" disambiguated with the language prefix "L".
        user_query = '"electric car" and L:EN'

    Reuters Instrument Code (RIC): MSFT.O: Obtains headlines for stories related to Microsoft company using Reuters Instrument Code (RIC)
        user_query = 'MSFT.O'

    Most Read News (M:1RS): Obtains headlines which have been most read.
        user_query = 'MRG'

    Daterange using LAST syntax (last 5 days): Obtains headlines for stories related to MRG written
        user_query = 'MRG last 5 days'

    Daterange with "from,to" explicit syntax: Obtains headlines for major breaking news stories being written in English and written between 2025-05-30 and 2025-05-31

    Parenthesis usage: Obtains headlines for stories about Korea which related to USA or China.
        user_query = 'Korea and (USA or China)'

    NOT operator: Obtains headlines for stories about "Debt" but not about the CMPNY.
        user_query = '"Debt" and NOT CMPNY'

    """
    search_url = f"{RDP_BASE_URL}/data/news/v1/headlines?query={user_query}"

    try:
        response = await make_authenticated_request(search_url, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        # Extract simplified story data
        simplified_stories = []
        if "data" in data:
            for story in data["data"]:
                story_data = {"story_id": story.get("storyId", ""), "headline": ""}

                # Extract headline from nested structure
                if (
                    "newsItem" in story
                    and "itemMeta" in story["newsItem"]
                    and "title" in story["newsItem"]["itemMeta"]
                ):
                    titles = story["newsItem"]["itemMeta"]["title"]
                    if titles and len(titles) > 0 and "$" in titles[0]:
                        story_data["headline"] = titles[0]["$"]

                simplified_stories.append(story_data)

        return json.dumps(simplified_stories)
    except Exception as e:
        return f"Error fetching news: {e}"


@mcp.tool()
async def get_news_story(storyId: str) -> str:
    """
    Retrieve detailed information about a specific news story using its unique identifier.

    This tool fetches comprehensive details about a news article, including headline, publication date,
    content type, and actual content. Use this after finding stories with query_news to get full details.

    Args:
        storyId (str): The unique story identifier from news system.
                      Format: 'urn:newsml:reuters.com:YYYY-MM-DD:nXXXXXXXX'
                      Example: 'urn:newsml:reuters.com:20250610:nL1N3SE0D8'
                      (Get this from the story_id field returned by query_news)

    Returns:
        str: JSON object containing detailed story information:
             - story_id: The unique identifier
             - headline: Full headline/title of the article
             - publication_date: ISO timestamp when the story was published
             - urgency: News urgency level (1=highest, 5=lowest priority)
             - content_type: Type of content (text/html, image/jpeg, etc.)
             - content: Main content (for images, notes binary data not included)
             - source: Information source code (e.g., NS:RTRS for Reuters)

    Example usage:
        1. First search: query_news("Tesla earnings")
        2. Then get details: get_news_story("urn:newsml:reuters.com:20250610:nL1N3SE0D8")

    Note: Some stories may be images, videos, or other media formats rather than text articles.
    """
    news_url = f"{RDP_BASE_URL}/data/news/v1/stories/{storyId}"

    try:
        response = await make_authenticated_request(news_url, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        # Extract simplified story information
        simplified_story = {
            "story_id": storyId,
            "headline": "",
            "publication_date": "",
            "urgency": "",
            "content_type": "",
            "content": "",
            "source": "",
        }

        if "newsItem" in data:
            news_item = data["newsItem"]

            # Extract headline
            if "contentMeta" in news_item and "headline" in news_item["contentMeta"]:
                headlines = news_item["contentMeta"]["headline"]
                if headlines and len(headlines) > 0 and "$" in headlines[0]:
                    simplified_story["headline"] = headlines[0]["$"]

            # Extract publication date
            if "itemMeta" in news_item and "versionCreated" in news_item["itemMeta"]:
                simplified_story["publication_date"] = news_item["itemMeta"][
                    "versionCreated"
                ]["$"]

            # Extract urgency
            if "contentMeta" in news_item and "urgency" in news_item["contentMeta"]:
                simplified_story["urgency"] = news_item["contentMeta"]["urgency"]["$"]

            # Extract source
            if "contentMeta" in news_item and "infoSource" in news_item["contentMeta"]:
                sources = news_item["contentMeta"]["infoSource"]
                if sources and len(sources) > 0:
                    simplified_story["source"] = sources[0].get("_qcode", "")

            # Check content type and extract relevant content
            if "contentSet" in news_item:
                content_set = news_item["contentSet"]
                if "inlineData" in content_set:
                    inline_data = content_set["inlineData"]
                    if inline_data and len(inline_data) > 0:
                        content_type = inline_data[0].get("_contenttype", "")
                        simplified_story["content_type"] = content_type

                        # For images, just note it's an image - don't include base64 data
                        if "image" in content_type:
                            simplified_story["content"] = (
                                "Image content (binary data not included)"
                            )
                        else:
                            # For text content, include the actual content
                            simplified_story["content"] = inline_data[0].get("$", "")

        return json.dumps(simplified_story)
    except Exception as e:
        return f"Error fetching news by ID: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
