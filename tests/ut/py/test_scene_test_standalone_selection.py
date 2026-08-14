# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone ``SceneTestCase.run_module`` class selection.

A scene-test module may import a ``SceneTestCase`` from another module (e.g.
the host_build_graph qwen3 decode imports the TMR class to reuse its
CALLABLE). Only classes *defined in* the run module are runnable targets:
running the file must not dispatch the imported class's runtime too.
"""

from __future__ import annotations

import sys
import textwrap
from types import ModuleType

from simpler_setup.scene_test import select_scene_classes


def _make_scene_module(name: str, imported: str | None = None) -> ModuleType:
    """Build a module whose classes carry ``__module__ == name``.

    ``imported`` optionally simulates `from other_module import OtherClass`
    binding a foreign SceneTestCase into the namespace.
    """
    module = ModuleType(name)
    # @scene_test resolves CALLABLE paths against inspect.getfile(cls), which
    # walks sys.modules[cls.__module__].__file__.
    module.__file__ = __file__
    sys.modules[name] = module
    import_block = f"from {imported} import TestForeignRuntime\n" if imported else ""
    code = textwrap.dedent(
        f"""
        from simpler_setup import SceneTestCase, scene_test

        {import_block}
        @scene_test(level=2, runtime="host_build_graph")
        class TestLocalHbg(SceneTestCase):
            CASES = [{{"name": "LocalCase", "platforms": ["a5sim"], "params": {{}}}}]

            def generate_args(self, params):
                raise NotImplementedError

            def compute_golden(self, args, params):
                raise NotImplementedError
        """
    )
    exec(code, vars(module))  # noqa: S102 -- test fixture, executes literal above
    return module


_FOREIGN = _make_scene_module("_foreign_tmr_module_stub")

# A stand-in for `examples.a5.tensormap_and_ringbuffer...TestQwen314BDecode`:
# a real SceneTestCase defined in a foreign module, meant to be imported by
# the module under test.
_FOREIGN_MODULE = ModuleType("_foreign_tmr_class_module")
_FOREIGN_MODULE.TestForeignRuntime = _FOREIGN.TestLocalHbg


def test_selects_only_classes_defined_in_module():
    sys.modules.setdefault("_foreign_tmr_class_module", _FOREIGN_MODULE)
    module = _make_scene_module("fake_scene_module", imported="_foreign_tmr_class_module")

    selected = select_scene_classes(module)

    assert [cls.__name__ for cls in selected] == ["TestLocalHbg"]
    assert selected[0]._st_runtime == "host_build_graph"


def test_select_scene_classes_empty_module():
    empty = ModuleType("empty_module")
    assert select_scene_classes(empty) == []
