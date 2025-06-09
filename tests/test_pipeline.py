import os
import sys
import asyncio
import pytest

# Ensure the project root is on the Python path so "pipeline" can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault('ZELIBOBA_TOKEN', 'dummy')

def test_process_prompt_triggers_missing_greenlet_error():
    from pipeline import process_prompt

    with pytest.raises(Exception) as excinfo:
        asyncio.run(process_prompt('p1', '{{ value }}'))

    assert 'greenlet_spawn' in str(excinfo.value)
