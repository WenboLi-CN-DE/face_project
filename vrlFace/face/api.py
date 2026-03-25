"""
人脸识别 FastAPI 路由

包含:
    /vrlFaceDetection     — 人脸检测
    /vrlFaceComparison    — 人脸 1:1 比对
    /vrlFaceSearch        — 人脸 1:N 搜索
    /vrlFaceIdComparison  — 人证比对（证件照 vs 现场照）
"""

import io
import logging
import traceback

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from .recognizer import face_detection, gen_verify_res, face_search, reload_face_db
from .config import config
from .id_preprocess import preprocess_id_photo

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def _read_image(upload: UploadFile, data: bytes) -> np.ndarray:
    """将上传文件字节解码为 RGB numpy 数组"""
    image = Image.open(io.BytesIO(data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def _check_content_type(upload: UploadFile) -> None:
    if upload.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{upload.content_type}，支持：{', '.join(_ALLOWED_TYPES)}",
        )


@router.post("/vrlFaceDetection")
async def vrl_face_detection(picture: UploadFile = File(...)):
    """
    人脸检测

    - **picture**: 图片文件 (JPEG / PNG / WEBP / BMP)
    """
    _check_content_type(picture)
    try:
        image = _read_image(picture, await picture.read())
        result = face_detection(image)
        return {
            "code": 0,
            "msg": "face detection successful",
            "detection_results": result,
            "filename": picture.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("vrlFaceDetection error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"识别过程出错：{str(e)}")


@router.post("/vrlFaceComparison")
async def vrl_face_comparison(
    picture1: UploadFile = File(...),
    picture2: UploadFile = File(...),
):
    """
    人脸 1:1 比对

    - **picture1**: 第一张图片
    - **picture2**: 第二张图片
    """
    _check_content_type(picture1)
    _check_content_type(picture2)
    try:
        image1 = _read_image(picture1, await picture1.read())
        image2 = _read_image(picture2, await picture2.read())
        result = gen_verify_res(image1, image2)
        return {
            "code": 0,
            "msg": "face comparison successful",
            "comparison_results": result,
            "filenames": [picture1.filename, picture2.filename],
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("vrlFaceComparison validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("vrlFaceComparison error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"识别过程出错：{str(e)}")


@router.post("/vrlFaceSearch")
async def vrl_face_search(picture: UploadFile = File(...)):
    """
    人脸 1:N 搜索

    - **picture**: 待搜索的图片
    """
    _check_content_type(picture)
    try:
        image = _read_image(picture, await picture.read())
        result = face_search(image, db_path=config.images_base, top_n=3)
        return {
            "code": 0,
            "msg": "face search successful",
            "search_results": result,
            "filename": picture.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("vrlFaceSearch error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"识别过程出错：{str(e)}")


@router.post("/vrlFaceSearch/reload")
async def vrl_face_search_reload():
    try:
        reload_face_db()
        from .recognizer import _face_db

        count = len(_face_db) if _face_db is not None else 0
        return {"code": 0, "msg": "face db reloaded successfully", "count": count}
    except Exception as e:
        logger.error("vrlFaceSearch reload error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"重新加载人脸库出错：{str(e)}")


@router.post("/vrlFaceIdComparison")
async def vrl_face_id_comparison(
    picture1: UploadFile = File(...),
    picture2: UploadFile = File(...),
):
    """
    人证比对（证件照 vs 现场人脸）

    将证件照与现场照进行 1:1 人脸比对，验证是否为同一人。
    返回的 confidence 为百分制（0~100），与 /vrlFaceComparison 的 0~1
    余弦相似度不同，便于前端直接展示。

    - **picture1**: 证件照（身份证、护照等）
    - **picture2**: 现场照 / 自拍
    """
    _check_content_type(picture1)
    _check_content_type(picture2)
    try:
        image1 = _read_image(picture1, await picture1.read())
        image2 = _read_image(picture2, await picture2.read())

        # 仅对证件照做预处理（对齐+裁剪+增强）
        image1 = preprocess_id_photo(image1)

        # 使用人证比对专用阈值（高于普通比对，安全性更强）
        result = gen_verify_res(
            image1, image2, threshold=config.id_comparison_threshold
        )

        # 将余弦相似度（0~1）转换为百分制，与方案文档保持一致
        result["confidence"] = round(result["confidence"] * 100, 2)

        return {
            "code": 0,
            "msg": "face comparison successful",
            "comparison_results": result,
            "filename": [picture1.filename, picture2.filename],
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("vrlFaceIdComparison validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("vrlFaceIdComparison error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"识别过程出错：{str(e)}")
