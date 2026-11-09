import numpy as np
import pytest

from studies.framework.core import SelectionBook


def test_selection_book_adds_parent_and_child_masks_without_mutating_parent():
    book = SelectionBook(4)
    base = book.add("beamline_quality", [True, False, True, True])
    child = book.add("prompt", [True, True, False, True], parent="beamline_quality")
    np.testing.assert_array_equal(base, [True, False, True, True])
    np.testing.assert_array_equal(child, [True, False, False, True])
    assert book.counts() == {"all": 4, "beamline_quality": 3, "prompt": 2}
    assert book.parent("prompt") == "beamline_quality"


def test_selection_book_rejects_replacement_unknown_parent_and_wrong_shape():
    book = SelectionBook(2)
    book.add("quality", [True, True])
    with pytest.raises(ValueError, match="must be new"):
        book.add("quality", [False, False])
    with pytest.raises(KeyError, match="Unknown parent"):
        book.add("child", [True, True], parent="missing")
    with pytest.raises(ValueError, match="one boolean value"):
        book.add("bad", [True])
