from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post('/test')
def testing():
    return {'message':'Its working fine'}

if __name__ == '__main__':
    uvicorn.run('demo:app', host='127.0.0.1',port=8000)
