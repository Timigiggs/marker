from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Iterable, List

from marker.builders import BaseBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class


class DocumentBuilder(BaseBuilder):
    """
    Constructs a Document given a PdfProvider, LayoutBuilder, and OcrBuilder.
    """
    lowres_image_dpi: Annotated[
        int,
        "DPI setting for low-resolution page images used for Layout and Line Detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False

    page_pipeline_chunk_size: Annotated[
        int,
        "The number of pages to batch together when running the page-level pipeline.",
    ] = 12
    page_pipeline_workers: Annotated[
        int,
        "The number of worker threads to use for the page pipeline stages.",
    ] = 3

    def __call__(
        self,
        provider: PdfProvider,
        layout_builder: LayoutBuilder,
        line_builder: LineBuilder,
        ocr_builder: OcrBuilder,
    ):
        document = self.build_document(provider)

        if not document.pages:
            return document

        # Fall back to sequential execution if pipeline disabled via configuration
        if self.page_pipeline_chunk_size <= 0 or self.page_pipeline_workers <= 0:
            self._load_images_for_pages(document.pages, provider)
            layout_builder.process_pages(document, provider, document.pages)
            layout_builder.expand_layout_blocks(document)
            line_builder.process_pages(document, provider, document.pages)
            if not self.disable_ocr:
                ocr_builder.process_pages(document, provider, document.pages)
            return document

        page_chunks = list(self._chunk_pages(document.pages, self.page_pipeline_chunk_size))

        def run_layout(pages: List[PageGroup]):
            layout_builder.process_pages(document, provider, pages)
            return pages

        def run_line(pages: List[PageGroup]):
            line_builder.process_pages(document, provider, pages)
            for page in pages:
                page.lowres_image = None
            return pages

        def run_ocr(pages: List[PageGroup]):
            ocr_builder.process_pages(document, provider, pages)
            return pages

        layout_futures = {}
        line_futures = {}
        ocr_futures = {}

        with ThreadPoolExecutor(max_workers=self.page_pipeline_workers) as executor:
            for idx, chunk_pages in enumerate(page_chunks):
                self._load_images_for_pages(chunk_pages, provider)
                layout_future = executor.submit(run_layout, chunk_pages)
                layout_futures[idx] = (layout_future, chunk_pages)

                prev_idx = idx - 1
                if prev_idx >= 0 and prev_idx in layout_futures and prev_idx not in line_futures:
                    prev_future, prev_pages = layout_futures[prev_idx]
                    prev_future.result()
                    line_futures[prev_idx] = (
                        executor.submit(run_line, prev_pages),
                        prev_pages,
                    )

                prev_prev_idx = idx - 2
                if (
                    not self.disable_ocr
                    and prev_prev_idx >= 0
                    and prev_prev_idx in line_futures
                    and prev_prev_idx not in ocr_futures
                ):
                    prev_line_future, prev_line_pages = line_futures[prev_prev_idx]
                    prev_line_future.result()
                    ocr_futures[prev_prev_idx] = executor.submit(
                        run_ocr, prev_line_pages
                    )

            # Ensure remaining stages are scheduled after the main loop completes
            for idx, (layout_future, pages) in layout_futures.items():
                if idx not in line_futures:
                    layout_future.result()
                    line_futures[idx] = (
                        executor.submit(run_line, pages),
                        pages,
                    )

            if not self.disable_ocr:
                for idx, (line_future, pages) in line_futures.items():
                    if idx not in ocr_futures:
                        line_future.result()
                        ocr_futures[idx] = executor.submit(run_ocr, pages)

            # Wait for all scheduled work to complete before returning
            for layout_future, _ in layout_futures.values():
                layout_future.result()
            for line_future, _ in line_futures.values():
                line_future.result()
            if not self.disable_ocr:
                for ocr_future in ocr_futures.values():
                    ocr_future.result()

        layout_builder.expand_layout_blocks(document)
        return document

    def build_document(self, provider: PdfProvider):
        PageGroupClass: PageGroup = get_block_class(BlockTypes.Page)
        initial_pages = [
            PageGroupClass(
                page_id=p,
                lowres_image=None,
                highres_image=None,
                polygon=provider.get_page_bbox(p),
                refs=provider.get_page_refs(p)
            ) for i, p in enumerate(provider.page_range)
        ]
        DocumentClass: Document = get_block_class(BlockTypes.Document)
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)

    def _chunk_pages(self, pages: List[PageGroup], chunk_size: int) -> Iterable[List[PageGroup]]:
        for idx in range(0, len(pages), chunk_size):
            yield pages[idx : idx + chunk_size]

    def _load_images_for_pages(self, pages: List[PageGroup], provider: PdfProvider):
        page_ids = [page.page_id for page in pages]
        lowres_images = provider.get_images(page_ids, self.lowres_image_dpi)
        highres_images = provider.get_images(page_ids, self.highres_image_dpi)
        for page, lowres_image, highres_image in zip(
            pages, lowres_images, highres_images
        ):
            page.lowres_image = lowres_image
            page.highres_image = highres_image
