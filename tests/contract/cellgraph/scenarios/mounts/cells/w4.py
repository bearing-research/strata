# @mount out file://@NB@/out rw
(out / "note.txt").write_text((out / "../data/n/one.txt").read_text())
note = (out / "note.txt").read_text()
