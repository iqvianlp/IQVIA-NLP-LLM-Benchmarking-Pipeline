from pipelines.benchmark_blurb_other_tasks.common import _get_annotated_text


class TestAddKeyAndAnnotatedText:

    def test_entities_order_same_as_in_text(self):
        insertions = {
            'offsets': [
                [1, 2],
                [3, 4]
            ],
            'type': [
                'TYPE1',
                'TYPE2'
            ]
        }
        raw_text = 'abcde'
        text = _get_annotated_text(insertions, raw_text)

        assert text == 'a@TYPE1$b@TYPE1$c@TYPE2$d@TYPE2$e'

    def test_entities_order_in_reverse(self):
        insertions = {
            'offsets': [
                [3, 4],
                [1, 2]
            ],
            'type': [
                'TYPE1',
                'TYPE2'
            ]
        }
        raw_text = 'abcde'
        text = _get_annotated_text(insertions, raw_text)

        assert text == 'a@TYPE2$b@TYPE2$c@TYPE1$d@TYPE1$e'

    def test_first_entity_at_start_of_text(self):
        insertions = {
            'offsets': [
                [0, 1],
                [3, 4]
            ],
            'type': [
                'TYPE1',
                'TYPE2'
            ]
        }
        raw_text = 'abcde'
        text = _get_annotated_text(insertions, raw_text)

        assert text == '@TYPE1$a@TYPE1$bc@TYPE2$d@TYPE2$e'

    def test_last_entity_at_end_of_text(self):
        insertions = {
            'offsets': [
                [1, 2],
                [4, 5]
            ],
            'type': [
                'TYPE1',
                'TYPE2'
            ]
        }
        raw_text = 'abcde'
        text = _get_annotated_text(insertions, raw_text)

        assert text == 'a@TYPE1$b@TYPE1$cd@TYPE2$e@TYPE2$'

    def test_adjacent_entities(self):
        insertions = {
            'offsets': [
                [1, 2],
                [2, 3]
            ],
            'type': [
                'TYPE1',
                'TYPE2'
            ]
        }
        raw_text = 'abcde'
        text = _get_annotated_text(insertions, raw_text)

        assert text == 'a@TYPE1$b@TYPE1$@TYPE2$c@TYPE2$de'

    def test_ddi_entities(self):
        insertions = {
            'offsets': [
                [1, 2],
                [2, 3]
            ],
            'type': [
                'TYPE1',
                'TYPE2'
            ]
        }
        raw_text = 'abcde'
        text = _get_annotated_text(insertions, raw_text, is_ddi=True)

        assert text == 'a@DRUG$b@DRUG$@DRUG$c@DRUG$de'
