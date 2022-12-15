import uvicorn
import io
from fastapi import FastAPI
from pydantic import BaseModel
from module import OwnGanModule
from fastapi.responses import Response

app = FastAPI()
instance = OwnGanModule()


class Body(BaseModel):
    category_a: str
    category_b: str


@app.get("/{category_a}/{category_b}")
async def read_root(category_a: str, category_b: str):
    result = instance.interpolate_own(category_a, category_b)
    img_byte_arr = io.BytesIO()
    result.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return Response(content=img_byte_arr, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8660)
