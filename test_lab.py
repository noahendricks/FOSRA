import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import httpx
    import asyncio

    async def test():
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:6333/collections")
            return resp
        
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
