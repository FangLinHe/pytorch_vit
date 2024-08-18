from pytorch_vit.transformer_encoder import TransformerEncoder


def test_construct_transformer_encoder():
    encoder = TransformerEncoder()
    assert encoder is not None
