"""
人脸识别核心模块 — 基于 InsightFace

提供:
    face_detection   — 人脸检测
    gen_verify_res   — 1:1 人脸比对
    face_search      — 1:N 人脸搜索
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path

from .config import config


# 全局单例
_recognizer = None


def get_recognizer():
    """获取或初始化人脸识别器（单例，使用全局配置）"""
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceAnalysis(name=config.model_name, providers=config.providers)
        _recognizer.prepare(ctx_id=config.ctx_id, det_size=config.det_size)
    return _recognizer


def detection_face_exits(img_path):
    """
    检测是否存在人脸

    Args:
        img_path: 图片路径或 numpy 数组

    Returns:
        (has_face, confidence): 是否有人脸和置信度
    """
    try:
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            if img is None:
                return False, 0.0
        else:
            img = img_path

        recognizer = get_recognizer()
        faces = recognizer.get(img)

        if len(faces) > 0:
            confidence = (
                float(faces[0].det_score) if hasattr(faces[0], "det_score") else 0.0
            )
            return True, confidence
        return False, 0.0

    except Exception as e:
        print(f"人脸检测失败：{e}")
        return False, 0.0


def verify_face(img1, img2, threshold=None):
    """
    人脸比对，验证是否为同一人

    Args:
        img1: 图片 1（路径或 numpy 数组）
        img2: 图片 2（路径或 numpy 数组）
        threshold: 相似度阈值（None 使用全局配置）

    Returns:
        (verified, similarity): 是否验证通过和相似度
    """
    if threshold is None:
        threshold = config.similarity_threshold

    try:
        if isinstance(img1, str):
            img1 = cv2.imread(img1)
        if isinstance(img2, str):
            img2 = cv2.imread(img2)

        recognizer = get_recognizer()
        faces1 = recognizer.get(img1)
        faces2 = recognizer.get(img2)

        if not faces1 or not faces2:
            return False, 0.0

        feat1 = faces1[0].embedding
        feat2 = faces2[0].embedding

        from numpy.linalg import norm

        similarity = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        verified = similarity >= threshold
        return verified, float(similarity)

    except Exception as e:
        print(f"人脸比对失败：{e}")
        return False, 0.0


def gen_verify_res(img1, img2, threshold=None):
    """
    生成 1:1 人脸比对结果

    Args:
        img1: 图片 1（路径或 numpy 数组）
        img2: 图片 2（路径或 numpy 数组）
        threshold: 相似度阈值（None 使用全局配置）

    Returns:
        dict: 比对结果字典
    """
    if threshold is None:
        threshold = config.similarity_threshold

    flag1, cnf1 = detection_face_exits(img1)
    flag2, cnf2 = detection_face_exits(img2)

    if flag1 and flag2:
        vrf, cnf_vrf = verify_face(img1, img2, threshold)
        vrf_flag = 1 if vrf else 0
        return {
            "is_face_exist": 1,
            "confidence_exist": [cnf1, cnf2],
            "is_same_face": vrf_flag,
            "confidence": cnf_vrf,
            "detection_result": "both pictures have face",
        }
    elif not flag1 and flag2:
        return {
            "is_face_exist": 0,
            "confidence_exist": [cnf1, cnf2],
            "detection_result": "picture 1 has no face",
            "is_same_face": -1,
            "confidence": 0.0,
        }
    elif flag1 and not flag2:
        return {
            "is_face_exist": 0,
            "confidence_exist": [cnf1, cnf2],
            "detection_result": "picture 2 has no face",
            "is_same_face": -1,
            "confidence": 0.0,
        }
    else:
        return {
            "is_face_exist": 0,
            "confidence_exist": [cnf1, cnf2],
            "detection_result": "both pictures have no face",
            "is_same_face": -1,
            "confidence": 0.0,
        }


def face_detection(img):
    """
    人脸检测

    Args:
        img: 图片路径或 numpy 数组

    Returns:
        dict: 检测结果
    """
    try:
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                return {"is_face_exist": 0, "face_num": 0, "faces_detected": []}

        recognizer = get_recognizer()
        faces = recognizer.get(img)

        if len(faces) > 0:
            res_lst = []
            for fc in faces:
                bbox = fc.bbox
                facial_area = {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2] - bbox[0]),
                    "height": int(bbox[3] - bbox[1]),
                }
                confidence = float(fc.det_score) if hasattr(fc, "det_score") else 0.0
                res_lst.append({"facial_area": facial_area, "confidence": confidence})

            return {
                "is_face_exist": 1,
                "face_num": len(res_lst),
                "faces_detected": res_lst,
            }

        return {"is_face_exist": 0, "face_num": 0, "faces_detected": []}

    except Exception as e:
        print(f"人脸检测失败：{e}")
        return {"is_face_exist": 0, "face_num": 0, "faces_detected": []}


def face_search(img, db_path=None, top_n=3):
    """
    1:N 人脸搜索 — 在数据库目录中搜索相似人脸

    Args:
        img: 待搜索的图片（路径或 numpy 数组）
        db_path: 数据库目录路径（None 使用全局配置 images_base）
        top_n: 返回最相似的 N 个结果

    Returns:
        dict: 搜索结果
    """
    if db_path is None:
        db_path = config.images_base

    try:
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                return {"searched_similar_pictures": [], "has_similar_picture": 0}

        recognizer = get_recognizer()
        faces = recognizer.get(img)
        if not faces:
            return {"searched_similar_pictures": [], "has_similar_picture": 0}

        query_embedding = faces[0].embedding
        db_path = Path(db_path)

        if not db_path.exists():
            return {"searched_similar_pictures": [], "has_similar_picture": 0}

        results = []
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        for img_file in db_path.iterdir():
            if img_file.suffix.lower() not in image_extensions:
                continue

            try:
                db_img = cv2.imread(str(img_file))
                if db_img is None:
                    continue

                db_faces = recognizer.get(db_img)
                if not db_faces:
                    continue

                db_embedding = db_faces[0].embedding

                from numpy.linalg import norm

                similarity = np.dot(query_embedding, db_embedding) / (
                    norm(query_embedding) * norm(db_embedding)
                )

                results.append(
                    {
                        "picture": str(img_file),
                        "confidence": float(similarity),
                        "distance": 1.0 - float(similarity),
                    }
                )

            except Exception as e:
                print(f"处理图片 {img_file} 失败：{e}")
                continue

        results.sort(key=lambda x: x["confidence"], reverse=True)
        top_results = results[:top_n]
        has_similar = (
            1
            if len(top_results) > 0 and top_results[0]["confidence"] > 0.5
            else 0
        )

        return {
            "searched_similar_pictures": top_results,
            "has_similar_picture": has_similar,
        }

    except Exception as e:
        print(f"人脸搜索失败：{e}")
        return {"searched_similar_pictures": [], "has_similar_picture": 0}

