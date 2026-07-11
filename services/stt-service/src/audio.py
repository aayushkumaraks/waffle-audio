from array import array


def pcm16_to_float_list(audio_bytes: bytes) -> list[float]:
    """
    Convert 16-bit signed PCM audio to normalized float samples.

    Input:
        bytes (little-endian PCM16)

    Output:
        list[float] in the range [-1.0, 1.0]
    """

    samples = array("h")
    samples.frombytes(audio_bytes)

    # needs to be optimized later
    return [sample / 32768.0 for sample in samples]