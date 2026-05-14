from langchain_community.tools import DuckDuckGoSearchRun

search_tool=DuckDuckGoSearchRun()

result = search_tool.invoke("can you give me a pitch report for todays match of ipl in raipur and will it help spinners ")

print(result)