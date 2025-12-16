from core.video_source import VideoSource

def test_video_source_open_close():
    vs = VideoSource(0)
    vs.open()
    frame = vs.read()
    assert frame is not None
    vs.release()
