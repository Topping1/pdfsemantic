import os
import sys
import pickle
import re
import numpy as np
import fitz
import requests
from PyQt5.QtCore import Qt, QSize, QRectF, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QKeySequence, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QLabel, QToolBar,
    QListWidget, QListWidgetItem, QHBoxLayout, QVBoxLayout, QWidget,
    QLineEdit, QPushButton, QSplitter, QSizePolicy, QShortcut, QScrollArea
)

# ───────── Embedding helpers (unchanged) ─────────
EMBED_URL, MODEL = "http://localhost:8080/v1/embeddings", "Qwen3-Embedding-0.6B"

def _norm(v, ctx):
    if v.size == 0 or np.isnan(v).any() or not np.linalg.norm(v):
        raise RuntimeError(f"Invalid embedding for {ctx}")
    return v / np.linalg.norm(v)

def get_embedding(txt: str):
    if not txt.endswith("<|endoftext|>"):
        txt = txt.rstrip() + " <|endoftext|>"
    vec = np.array(requests.post(EMBED_URL,
              json={"model": MODEL, "input": txt}, timeout=30)
              .json()["data"][0]["embedding"], dtype=np.float32)
    return _norm(vec, "query")

# ───────── Chunk helpers (FIXED) ─────────
sent_re = re.compile(r"(?<=[.!?])\s+")

def sentence_split(t):
    return sent_re.split(t.strip())

def chunk_page(txt, tgt):
    """
    Splits text into chunks around a target size `tgt`.
    FIX: Handles sentences that are individually longer than `tgt` by breaking them up.
    """
    out, buf, ln = [], [], 0
    for s in sentence_split(txt.strip()):
        s_len = len(s)
        if s_len == 0:
            continue

        if ln + s_len > tgt and buf:
            out.append(" ".join(buf))
            buf, ln = [], 0

        if s_len > tgt:
            if buf:
                out.append(" ".join(buf))
                buf, ln = [], 0
            for i in range(0, s_len, tgt):
                out.append(s[i:i+tgt])
        else:
            buf.append(s)
            ln += s_len

    if buf:
        out.append(" ".join(buf))
    final_chunks = [c for c in out if c]
    return final_chunks, len(final_chunks)

# ───────── Worker thread (unchanged) ─────────
class EmbedWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(list, list, object)

    def __init__(self, pdf):
        super().__init__()
        self.path = pdf

    def run(self):
        doc = fitz.open(self.path)
        pages = doc.page_count
        avg = sum(len(doc.load_page(i).get_text("text")) for i in range(pages)) / pages if pages > 0 else 0
        tgt = max(int(avg / 4), 400)
        txts, chunks, vecs = [], [], []
        for p in range(pages):
            t = doc.load_page(p).get_text("text")
            txts.append(t)
            if not t.strip():
                continue
            chs, tot = chunk_page(t, tgt)
            for i, ck in enumerate(chs):
                chunks.append({"page": p, "text": ck,
                               "band": i, "bands_total": tot})
                vecs.append(get_embedding(ck))
            self.progress.emit(p+1, pages)
        self.done.emit(txts, chunks, np.stack(vecs) if vecs else np.array([]))

# ───────── GUI ─────────
class Viewer(QMainWindow):
    ZS, ZMIN, ZMAX = 0.25, 0.25, 4.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Semantic Viewer")
        self.resize(1200, 800)
        self.statusBar()

        # state
        self.pdf = None
        self.page_cache = {}
        self.page = self.zoom = 0
        self.page_texts = []
        self.chunks = []
        self.emb = None
        self.hl_page = self.hl_rects = self.band_meta = None
        self.worker = None
        self.cache = None

        # widgets
        self.img = QLabel(alignment=Qt.AlignCenter)
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll = QScrollArea(widgetResizable=True)
        self.scroll.setWidget(self.img)

        self.results = QListWidget(wordWrap=True)
        self.results.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.q = QLineEdit(placeholderText="Type query ↵")
        self.btn = QPushButton("Search", default=True)
        self.q.returnPressed.connect(self.btn.click)
        top = QWidget()
        hb = QHBoxLayout(top)
        hb.setContentsMargins(2, 2, 2, 2)
        hb.addWidget(self.q)
        hb.addWidget(self.btn)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(top)
        rv.addWidget(self.results)

        self.split = QSplitter()
        self.split.addWidget(self.scroll)
        self.split.addWidget(right)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 1)
        QTimer.singleShot(0, lambda: self.split.setSizes([self.width()-300, 300]))
        self.setCentralWidget(self.split)

        # toolbar
        tb = QToolBar()
        tb.setIconSize(QSize(24, 24))
        self.addToolBar(tb)

        # Add icons to the toolbar actions
        def add_action(name, icon, fn):
            action = QAction(QIcon.fromTheme(icon), name, self)
            action.triggered.connect(fn)
            tb.addAction(action)

        add_action("Open", "document-open", self.open_pdf)
        add_action("First", "go-first", lambda: self.goto(0))
        add_action("Prev", "go-previous", lambda: self.goto(self.page-1))
        add_action("Next", "go-next", lambda: self.goto(self.page+1))
        add_action("Last", "go-last", lambda: self.goto(self.pdf.page_count-1 if self.pdf else 0))
        add_action("Zoom +", "zoom-in", self.zin)
        add_action("Zoom –", "zoom-out", self.zout)

        # shortcuts
        for k, fn in ((Qt.Key_PageUp, lambda: self.goto(self.page-1)),
                      (Qt.Key_PageDown, lambda: self.goto(self.page+1)),
                      (Qt.Key_Home, lambda: self.goto(0)),
                      (Qt.Key_End, lambda: self.goto(self.pdf.page_count-1 if self.pdf else 0)),
                      (QKeySequence.ZoomIn, self.zin),
                      (QKeySequence.ZoomOut, self.zout)):
            QShortcut(QKeySequence(k), self).activated.connect(fn)

        # signals
        self.btn.clicked.connect(self.search)
        self.results.itemClicked.connect(self.activate)
        self.results.currentRowChanged.connect(lambda r: r >= 0 and self.activate(self.results.item(r)))

    # ---- File I/O ----
    def open_pdf(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF (*.pdf)")
        if p:
            self.load(p)

    def load(self, path):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.worker = None
        self.statusBar().clearMessage()
        self.pdf = fitz.open(path)
        self.page_cache.clear()
        self.page = 0
        self.zoom = 1.0
        self.cache = os.path.splitext(path)[0] + ".em"
        self.goto(0, True)
        self.results.clear()  # Clear the results listbox when loading a new PDF

        if self.load_cache():
            self.statusBar().showMessage("Embeddings loaded", 3000)
        else:
            self.statusBar().showMessage("Embedding … 0%")
            self.worker = EmbedWorker(path)
            self.worker.progress.connect(lambda d, t: self.statusBar()
                                          .showMessage(f"Embedding … {int(d/t*100)}%"))
            self.worker.done.connect(self.finish_embed)
            self.worker.finished.connect(lambda: setattr(self, "worker", None))
            self.worker.start()

    def load_cache(self):
        if not os.path.exists(self.cache):
            return False
        try:
            pt, ch, emb = pickle.load(open(self.cache, "rb"))
            if emb.size > 0 and (np.isnan(emb).any() or not np.linalg.norm(emb[0])):
                raise ValueError
            self.page_texts, self.chunks, self.emb = pt, ch, emb
            return True
        except:
            try:
                os.remove(self.cache)
            except OSError:
                pass
            return False

    def finish_embed(self, pt, ch, emb):
        self.page_texts, self.chunks, self.emb = pt, ch, emb
        if emb.size > 0:
            try:
                with open(self.cache, "wb") as f:
                    pickle.dump((pt, ch, emb), f)
                self.statusBar().showMessage("Embedding finished", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"Error saving cache: {e}", 5000)
        else:
            self.statusBar().showMessage("Embedding failed: no content found.", 5000)

    # ---- Navigation / zoom ----
    def goto(self, i, force=False):
        if not self.pdf:
            return
        i = max(0, min(i, self.pdf.page_count-1))
        if force or i != self.page:
            self.page = i
            self.render()

    def zin(self):
        self.zoom = min(self.zoom+self.ZS, self.ZMAX)
        self.render()

    def zout(self):
        self.zoom = max(self.zoom-self.ZS, self.ZMIN)
        self.render()

    # ---- Rendering (PATCHED) ----
    def render(self):
        if not self.pdf:
            return
        # Use a zoom factor of 1.0 for the initial rendering if not set
        if self.zoom == 0:
            self.zoom = 1.0
        key = (self.page, self.zoom)
        if key not in self.page_cache:
            pg = self.pdf.load_page(self.page)
            pix = pg.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))

            # Create a deep copy of the pixel data.
            # The QImage is constructed from this copy, ensuring it has its own
            # data and is not dependent on the lifetime of the 'pix' object.
            buf = bytes(pix.samples)

            fmt = QImage.Format_RGBA8888 if pix.alpha else QImage.Format_RGB888
            img = QImage(buf, pix.width, pix.height, pix.stride, fmt)
            self.page_cache[key] = QPixmap.fromImage(img)

        pm = QPixmap(self.page_cache[key])
        if self.hl_page == self.page:
            p = QPainter(pm)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 0, 120))
            if self.hl_rects:
                for r in self.hl_rects:
                    p.drawRect(QRectF(r.x0*self.zoom, r.y0*self.zoom,
                                      (r.x1-r.x0)*self.zoom, (r.y1-r.y0)*self.zoom))
            else:
                b, t = self.band_meta
                h = pm.height()/t
                p.drawRect(QRectF(0, b*h, pm.width(), h))
            p.end()
        self.img.setPixmap(pm)

    # ---- Search / activate ----
    def search(self):
        if self.emb is None or self.emb.size == 0:
            self.statusBar().showMessage("Embeddings not ready or empty", 3000)
            return
        q = self.q.text().strip()
        if not q:
            return
        sims = self.emb @ get_embedding(q)
        sims = np.nan_to_num(sims, nan=-np.inf)
        top = np.argsort(-sims)[:20]
        self.results.clear()
        for r, idx in enumerate(top, 1):
            c = self.chunks[idx]
            snippet = c["text"][:120].replace("\n", " ")
            it = QListWidgetItem(f"{r:02d}. p.{c['page']+1} – {snippet}…")
            it.setData(Qt.UserRole, idx)
            self.results.addItem(it)
        if self.results.count():
            self.results.setCurrentRow(0)

    def activate(self, item):
        if self.emb is None:
            return
        idx = item.data(Qt.UserRole)
        c = self.chunks[idx]
        pg = self.pdf.load_page(c["page"])
        text_instances = pg.search_for(c["text"])
        if text_instances:
            self.hl_rects = text_instances
        else:
            self.hl_rects = []
        self.hl_page = c["page"]
        self.band_meta = (c["band"], c["bands_total"])
        self.page_cache.pop((self.hl_page, self.zoom), None)
        self.goto(self.hl_page, True)

# ───────── main ─────────
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    Viewer().show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
