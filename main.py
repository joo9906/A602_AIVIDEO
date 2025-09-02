from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import route

app = FastAPI()

# CORS 설정 (프론트와 통신 위해 필수)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인 지정 예정.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(route.router)