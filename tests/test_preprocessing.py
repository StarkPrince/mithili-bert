from mithili_bert.preprocessing import split_sentences


def test_split_sentences_handles_devanagari_punctuation():
    text = "पहिल वाक्य। दोसर वाक्य॥ तेसर वाक्य? चारिम वाक्य!"

    assert split_sentences(text) == [
        "पहिल वाक्य।",
        "दोसर वाक्य॥",
        "तेसर वाक्य?",
        "चारिम वाक्य!",
    ]

