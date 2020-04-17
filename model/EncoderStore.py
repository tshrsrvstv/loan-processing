class EncoderStore():
    encoder_data = {}

    @classmethod
    def save(cls, key, encoder_object):
        EncoderStore.encoder_data[key] = encoder_object

    @classmethod
    def get(cls, key):
        return EncoderStore.encoder_data[key]
