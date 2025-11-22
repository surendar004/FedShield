import pytest
from pathlib import Path

def test_project_structure():
    """Test if all required directories exist"""
    root = Path(__file__).parent.parent
    required_dirs = ['server', 'client', 'dashboard', 'data', 'scripts']
    
    for dir_name in required_dirs:
        assert (root / dir_name).is_dir(), f"{dir_name} directory is missing"

def test_python_environment():
    """Test if all required packages are installed"""
    required_packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'flask',
        'streamlit',
        'plotly',
        'requests',
        'joblib',
        'flwr'
    ]
    
    from importlib.metadata import distribution, PackageNotFoundError
    
    for package in required_packages:
        try:
            distribution(package)
        except PackageNotFoundError:
            assert False, f"{package} is not installed"