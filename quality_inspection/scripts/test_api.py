"""
Test script for Quality Inspector API.
Verifies that the FastAPI endpoint works correctly.
"""

import requests
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_inspect_image():
    """Test image inspection endpoint."""
    print("\nTesting /inspect/image endpoint...")

    # Use a test image from the dataset
    test_image_path = PROJECT_ROOT / "data" / "raw" / "autovi" / "bracket_assembly" / "test" / "good" / "000.png"

    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return False

    with open(test_image_path, "rb") as f:
        files = {"file": f}
        data = {
            "camera_id": "test_camera",
            "product_id": "test_product",
            "save_evidence": "true",
        }

        response = requests.post(f"{API_URL}/inspect/image", files=files, data=data)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Event ID: {result['event_id']}")
        print(f"Decision: {result['decision']}")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_stats():
    """Test stats endpoint."""
    print("\nTesting /stats endpoint...")
    response = requests.get(f"{API_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


if __name__ == "__main__":
    print("=" * 60)
    print("Quality Inspector API Test Suite")
    print("=" * 60)

    try:
        # Test health
        health_ok = test_health()

        if not health_ok:
            print("\n[ERROR] API is not running. Start it with:")
            print("  cd quality_inspection && python3 -m uvicorn api.main:app --reload")
            sys.exit(1)

        # Test image inspection
        image_ok = test_inspect_image()

        # Test stats
        stats_ok = test_stats()

        print("\n" + "=" * 60)
        if health_ok and image_ok and stats_ok:
            print("[SUCCESS] All API tests passed")
        else:
            print("[WARNING] Some tests failed")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to API. Start it with:")
        print("  cd quality_inspection && python3 -m uvicorn api.main:app --reload")
        sys.exit(1)
