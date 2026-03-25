import pytest
import numpy as np
from pathlib import Path
from vrlFace.face import recognizer
from vrlFace.face.config import config

def test_get_face_db_returns_dict(tmp_path):
    """Test get_face_db returns a dictionary of features"""
    # Create some dummy image files
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "test1.jpg").touch()
    
    # Save old config and monkeypatch
    old_base = config.images_base
    config.images_base = str(img_dir)
    
    try:
        # Should be callable and return dict
        db = recognizer.get_face_db()
        assert isinstance(db, dict)
    finally:
        config.images_base = old_base

def test_reload_face_db(tmp_path):
    """Test reload_face_db updates the memory DB"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Save old config and monkeypatch
    old_base = config.images_base
    config.images_base = str(img_dir)
    
    try:
        # DB should be empty
        db = recognizer.get_face_db()
        assert len(db) == 0
        
        # Add a dummy file
        (img_dir / "test2.jpg").touch()
        
        # Before reload, db is cached (still 0 since we didn't add logic to mock extraction yet, but size checking is enough)
        # Actually since recognizer.get() will fail on empty files, the db will remain empty, so we patch recognizer
        class DummyRecognizer:
            def get(self, img):
                class DummyFace:
                    embedding = np.array([0.1, 0.2, 0.3])
                return [DummyFace()]
                
        import cv2
        old_imread = cv2.imread
        cv2.imread = lambda x: np.zeros((10,10,3))
        
        old_get_recognizer = recognizer.get_recognizer
        recognizer.get_recognizer = lambda: DummyRecognizer()
        
        try:
            recognizer.reload_face_db()
            db = recognizer.get_face_db()
            assert len(db) == 1
            assert str(img_dir / "test2.jpg") in db
        finally:
            cv2.imread = old_imread
            recognizer.get_recognizer = old_get_recognizer
            
    finally:
        config.images_base = old_base

def test_face_search_uses_memory_db(monkeypatch):
    """Test face_search uses memory DB directly without reading files again"""
    # Create fake DB in memory
    fake_db = {
        "fake_img1.jpg": np.array([1.0, 0.0, 0.0]),
        "fake_img2.jpg": np.array([0.0, 1.0, 0.0])
    }
    
    # Inject fake DB
    import vrlFace.face.recognizer
    monkeypatch.setattr(vrlFace.face.recognizer, "_face_db", fake_db)
    
    # Mock image read and recognizer
    class DummyFace:
        embedding = np.array([0.9, 0.1, 0.0])
        
    class DummyRecognizer:
        def get(self, img):
            return [DummyFace()]
            
    monkeypatch.setattr(vrlFace.face.recognizer, "get_recognizer", lambda: DummyRecognizer())
    
    # Create dummy image to search
    test_img = np.zeros((10,10,3), dtype=np.uint8)
    
    # Use fake path to skip existence checks
    from pathlib import Path
    class FakePath:
        def exists(self): return True
        def iterdir(self): return []
        def __str__(self): return "fake_path"
        
    monkeypatch.setattr(vrlFace.face.recognizer, "Path", lambda x: FakePath())
    
    # Expected config behavior
    monkeypatch.setattr(config, "similarity_threshold", 0.5)
    
    result = recognizer.face_search(test_img)
    
    assert result["has_similar_picture"] == 1
    assert len(result["searched_similar_pictures"]) > 0
    assert result["searched_similar_pictures"][0]["picture"] == "fake_img1.jpg"
