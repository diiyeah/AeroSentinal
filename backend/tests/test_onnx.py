"""
AeroSentinal — ONNX Integration Tests
======================================
Verifies that all ONNX models are loaded and used during inference,
and that outputs carry `is_synthetic_data=False`.
"""

import pytest
from backend.app.services.inference import (
    EngineService,
    HydraulicsService,
    LandingGearService,
    APUService,
)

def test_models_loaded_status():
    """Assert that the model files were found and loaded successfully."""
    assert EngineService.MODEL_LOADED is True, "Engine ONNX model is not loaded"
    assert HydraulicsService.MODEL_LOADED is True, "Hydraulics ONNX model is not loaded"
    assert LandingGearService.MODEL_LOADED is True, "Landing Gear ONNX model is not loaded"
    assert APUService.MODEL_LOADED is True, "APU ONNX model is not loaded"

def test_engine_inference_non_synthetic():
    pred = EngineService.predict(unit_id=1, cycle=50, sensors={})
    assert pred.is_synthetic_data is False
    assert pred.predicted_rul > 0

def test_hydraulics_inference_non_synthetic():
    pred = HydraulicsService.predict(sensors={"PS1": 160.0})
    assert pred.is_synthetic_data is False
    assert pred.anomaly_detected is False

def test_landing_gear_inference_non_synthetic():
    pred = LandingGearService.predict(brake_wear_pct=30.0)
    assert pred.is_synthetic_data is False
    assert pred.wear_severity in ("nominal", "moderate", "severe", "critical")

def test_apu_inference_non_synthetic():
    pred = APUService.predict(fouling_factor=0.1)
    assert pred.is_synthetic_data is False
    assert pred.health_score > 80
