import csv
import json
from io import StringIO
from typing import Annotated, Any, List
from openai import OpenAI
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import instructor
import pandas as pd


client = instructor.patch(OpenAI(), mode=instructor.function_calls.Mode.MD_JSON)

def save_tables_to_csv(tables, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for table in tables:
            writer.writerow([table.caption])
            writer.writerows(table.dataframe.values.tolist())
            writer.writerow([])  # Add empty rows between tables
            writer.writerow([])
            writer.writerow([])

class TableEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Table):
            cleaned_data = {}
            for key, value in o.dataframe.to_dict().items():
                cleaned_key = key.strip()
                cleaned_values = {sub_key.strip(): sub_value for sub_key, sub_value in value.items()}
                cleaned_data[cleaned_key] = cleaned_values
            return {
                "caption": o.caption,
                "dataframe": cleaned_data,
            }
        return super().default(o)


def save_tables_to_json(tables, filename):
    with open(filename, "w", encoding='utf-8') as jsonfile:
        json.dump(tables, jsonfile, indent=4, cls=TableEncoder)

def md_to_df(data: Any) -> Any:
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Get rid of whitespaces
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .map(lambda x: x.strip())
        )
    return data


MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda x: x.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": """
                The markdown representation of the table,
                each one should be tidy, do not try to join tables
                that should be seperate""",
        }
    ),
]


class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame


class MultipleTables(BaseModel):
    tables: List[Table]


example = MultipleTables(
    tables=[
        Table(
            caption="This is a caption",
            dataframe=pd.DataFrame(
                {
                    "Chart A": [10, 40],
                    "Chart B": [20, 50],
                    "Chart C": [30, 60],
                }
            ),
        )
    ]
)


def extract(url: str) -> MultipleTables:
    tables = client.chat.completions.create(
        model="gpt-4-vision-preview",
        max_tokens=4000,
        response_model=MultipleTables,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this data accurately as a table in markdown format. {example.model_dump_json(indent=2)}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                    {
                        "type": "text",
                        "text": """
                            First take a moment to reason about the best set of headers for the tables.
                            Write a good h1 for the image above. Then follow up with a short description of the what the data is about.
                            Then for each table you identified, write a h2 tag that is a descriptive title of the table.
                            Then follow up with a short description of the what the data is about.
                            Lastly, produce the markdown table for each table you identified.
                            Make sure to escape the markdown table properly, and make sure to include the caption and the dataframe.
                            including escaping all the newlines and quotes. Only return a markdown table in dataframe, nothing else.
                        """,
                    },
                ],
            }
        ],
    )
    print(type(tables))
    return tables


if __name__ == "__main__":
    urls = [
        "https://s3.documentcloud.org/documents/24428888/pages/image-25-p1-large.gif",
    ]
    for url in urls:
        tables = extract(url)
        save_tables_to_json(tables.tables, "test.json")
        #for table in tables.tables:
            #print(table.caption)
            #print(table.dataframe)
