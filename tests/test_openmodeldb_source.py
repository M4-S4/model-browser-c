import asyncio
import logging
import sys

import pytest

from backend.cache import Cache
from backend.http import SourceHttp, SourceInvalidRequest, SourceNotFound, SourceUnavailable
from backend.security import DOWNLOAD_HOSTS, html_to_text, is_allowed_download_url
from backend.sources import SearchQuery
from backend.sources import openmodeldb as omdb
from backend.sources.openmodeldb import (
    CATALOG_TTL_SECONDS,
    FAILURE_HOLD_SECONDS,
    RETRY_SECONDS,
    OpenModelDBSource,
    ParsedQuery,
    build_catalog,
    compatibility_note,
    description_html,
    facts_line,
    file_name,
    is_direct_url,
    license_from,
    model_files,
    normalize_detail,
    normalize_item,
    other_downloads,
    parse_model_url,
    parse_query,
)
from backend.tables import create_tables

from .civitai_fakes import MemoryCache, fake_settings
from .openmodeldb_fakes import (
    ARCHITECTURES,
    BACKBLAZE,
    GITHUB_RAW,
    GITHUB_RELEASE,
    GOOGLE_DRIVE_ONLY,
    GRAYSCALE,
    HTTP_ONLY,
    HUGGING_FACE,
    INLINE_HTML,
    MEGA_ONLY,
    MISSPELT_KEYS,
    MIXED,
    MODELS,
    NO_LICENSE,
    ONE_NAME_LIST,
    ONNX_AND_DRIVE,
    ONNX_FIRST,
    ONNX_ONLY,
    ORACLE,
    PAIR_WITHOUT_SIZES,
    PTH_TAR,
    RAW_GITHUB,
    STANDALONE_IMAGES,
    TAGS,
    THREE_FORMATS,
    TWO_AUTHORS,
    VIDEO,
    ZIP_ONLY,
    OpenModelDBTransport,
    catalog,
    make_source,
    model,
    models,
)

_SANITISED_TAGS = {"p", "br", "b", "strong", "i", "em", "u", "s", "ul", "ol", "li", "blockquote", "code", "pre", "hr", "a", "h3", "h4"}
NEWEST = ["1x-Archiver-AntiLines", "1x-Book-Compact", "4x-UltraSharpV2", "2x-LiveActionV1-SPAN", "2x-GameUp-TSCUNet"]
DOWNLOADABLE = 17
NOTE = (
    "Note: PotionUI's built-in upscaler currently loads ESRGAN (.pth) models only; "
    "this {}model will download and index but may not run in PotionUI."
)
T0 = 1_800_000_000.0


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    return fake_settings(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_render_memo():
    omdb._render_memo.clear()
    yield
    omdb._render_memo.clear()


@pytest.fixture
def clock():
    class Clock:
        now = T0

        def __call__(self):
            return self.now

    return Clock()


def _detail(model_id, cat=None):
    cat = cat or catalog()
    return normalize_detail(model_id, cat.models[model_id], cat)


def _item(model_id, cat=None):
    cat = cat or catalog()
    return normalize_item(model_id, cat.models[model_id], cat)


def _alone(model_id, raw):
    return catalog({model_id: raw})


def _ids(page):
    return [found.id for found in page.items]


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth?download=true",
        "https://github.com/Loganavter/Archivist-Project-Denoiser/releases/download/v1.0/1x-Archivist_AntiLines.pth",
        "https://github.com/Phhofm/models/raw/main/4xNomos8kSC/4xNomos8kSC.pth",
        "https://objects.githubusercontent.com/github-production-release-asset-2e65be/1/x.pth",
        "https://raw.githubusercontent.com/terrainer/AI-Upscaling-Models/main/4xSPANkendata/4xSPANkendata_fp32.onnx",
        "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/16x-ESRGAN.pth",
        "https://f002.backblazeb2.com/file/ESRGAN/_RELEASE/4x_Fatality_01_375000_G.pth",
    ],
)
def test_direct_urls(url):
    assert is_direct_url(url)
    assert is_allowed_download_url("openmodeldb", url)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/Kim2091/UltraSharpV2/blob/main/4x-UltraSharpV2_fp32_op17.onnx",
        "https://huggingface.co/Kim2091/UltraSharpV2",
        "https://github.com/renarchi/Re-SISR/releases/tag/Adore",
        "https://github.com/Venomalia/HDcube/tree/main/Compact#hdcube-compact",
        "https://github.com/Phhofm/models/blob/main/2xHFA2kAVCSRFormer_light/onnx/x.onnx",
        "https://github.com/Phhofm/models",
        "https://github.com/raw/main/x.pth",
        "https://github.com/releases/download/v1/x.pth",
        "https://drive.google.com/file/d/1abc/view?usp=sharing",
        "https://mega.nz/file/tWEECawD#PG3_dUsHQ",
        "https://1drv.ms/u/s!AbC",
        "https://www.mediafire.com/file/abc/model.pth/file",
        "https://u.pcloud.link/publink/show?code=abc",
        "https://drive.proton.me/urls/ABC#def",
        "https://icedrive.net/1/43GNBihZyi",
        "https://1fichier.com/?abc",
        "https://www.dropbox.com/s/52hsqa7ymt5pgga/KDM003_scans_1x.pth",
        "http://mufunyo.net/esrgan/4x_AnalogFrames_1.0_G.pth",
        "http://objectstorage.us-phoenix-1.oraclecloud.com/n/a/b/c/o/x.pth",
        "https://oraclecloud.com/x.pth",
        "https://backblazeb2.com/file/x.pth",
        "https://cdn-lfs.huggingface.co/repos/x",
        "https://user:pass@f002.backblazeb2.com/file/x.pth",
        "https://f002.backblazeb2.com:8443/file/x.pth",
        "https://127.0.0.1/x.pth",
        "https://f002.backblazeb2.com/file/x y.pth",
        None,
        "",
    ],
)
def test_pages_and_other_hosts_are_not_direct(url):
    assert not is_direct_url(url)


def test_the_allowlist_takes_every_direct_host_and_nothing_else():
    assert DOWNLOAD_HOSTS["openmodeldb"] == {
        "huggingface.co",
        "github.com",
        "objects.githubusercontent.com",
        "raw.githubusercontent.com",
        "*.oraclecloud.com",
        "*.backblazeb2.com",
    }
    files = [file for model_id in models() for file in _detail(model_id).versions[0].files]
    assert files and all(is_allowed_download_url("openmodeldb", file.download_url) for file in files)


def test_archives_are_not_the_file_they_contain():
    for model_id in (ZIP_ONLY, PTH_TAR):
        detail = _detail(model_id)
        assert (detail.mb_type, detail.versions[0].files) == (None, []), model_id
    assert other_downloads(model(PTH_TAR)) == ["GitHub"]
    assert other_downloads(model(ZIP_ONLY)) == ["ONNX"]


def test_file_names():
    assert file_name("4x-X", "https://h.example/o/a%20b.pth", "pth") == "a b.pth"
    assert file_name("4x-X", "https://h.example/o/a.PTH?download=true", "pth") == "a.PTH"
    assert file_name("4x-X", "https://h.example/o/", "pth") == "4x-X.pth"
    assert file_name("4x-X", "https://h.example/o/x%2Fy.pth", "pth") == "4x-X.pth"
    assert file_name("4x-X", "https://h.example/o/c%3A.pth", "pth") == "4x-X.pth"
    assert file_name("4x-X", "https://h.example/o/model", None) == "model"
    assert file_name("4x-X", "https://h.example/o/", "../weird") == "4x-X"


def test_an_oracle_model_is_an_upscaler_with_its_primary_hash():
    item = _item(ORACLE)

    assert item.model_dump() == {
        "source": "openmodeldb",
        "id": "16x-ESRGAN",
        "name": "16xESRGAN",
        "type": "ESRGAN",
        "mb_type": "upscaler",
        "base_model": "ESRGAN 16x",
        "base_models": ["ESRGAN 16x"],
        "creator": "victorca25",
        "nsfw": False,
        "poi": False,
        "early_access": False,
        "gated": False,
        "thumbnail": None,
        "stats": {"downloads": None, "likes": None, "rating": None},
        "source_url": "https://openmodeldb.info/models/16x-ESRGAN",
        "primary_sha256": "8701acac21c4c412a3e05242b389700c1248ff52314380793c045326f4c61dfb",
        "installed": None,
    }


@pytest.mark.parametrize(
    "model_id", [GOOGLE_DRIVE_ONLY, MEGA_ONLY, HTTP_ONLY, ZIP_ONLY, STANDALONE_IMAGES, ONNX_ONLY, ONNX_AND_DRIVE]
)
def test_a_model_without_a_file_to_offer_is_not_downloadable(model_id):
    item = _item(model_id)

    assert (item.mb_type, item.primary_sha256) == (None, None)
    assert item.type and item.base_model and item.source_url.endswith(model_id)


def test_every_item_is_safe_and_every_image_level_1():
    cat = catalog()
    for model_id in cat.models:
        item, detail = _item(model_id, cat), _detail(model_id, cat)
        assert not (item.nsfw or item.poi or item.early_access or item.gated or detail.nsfw or detail.poi)
        images = detail.versions[0].images + ([item.thumbnail] if item.thumbnail else [])
        assert all(image.nsfw is False and image.nsfw_level == 1 and not image.poi for image in images)


def test_authors_tags_and_scales():
    assert _item(TWO_AUTHORS).creator == "skr, zarxrax"
    assert _item(ONE_NAME_LIST).creator == "jingyunliang"
    assert (_item(ONE_NAME_LIST).base_model, _item(VIDEO).base_model) == ("SwinIR 8x", "CAIN 2x")
    assert _detail(GITHUB_RAW).tags == ["Compression Removal", "JPEG", "Realistic", "Restoration"]


def test_without_tags_or_architectures_files_the_keys_are_the_labels():
    cat = catalog(tags=False, architectures=False)

    item = _item(GITHUB_RAW, cat)

    assert (item.type, item.base_model) == ("esrgan", "esrgan 4x")
    assert _detail(GITHUB_RAW, cat).tags == ["compression-removal", "jpeg", "photo", "restoration"]
    assert facts_line(cat.models[GITHUB_RAW], cat).startswith("Architecture: esrgan · Scale: 4x · Channels: 3 → 3")


def test_an_entry_missing_its_architecture_and_scale_still_normalises():
    raw = model(ORACLE)
    del raw["architecture"], raw["scale"], raw["date"], raw["author"], raw["name"]

    item = _item(ORACLE, _alone(ORACLE, raw))

    assert (item.type, item.base_model, item.base_models) == ("Upscaler", None, [])
    assert (item.name, item.creator) == (ORACLE, None)


def test_detail_of_an_oracle_model():
    detail = _detail(ORACLE)

    assert (detail.source, detail.id, detail.name, detail.type, detail.mb_type) == (
        "openmodeldb", "16x-ESRGAN", "16xESRGAN", "ESRGAN", "upscaler",
    )
    assert detail.license.model_dump() == {
        "summary": "Apache-2.0",
        "commercial_use": None,
        "no_credit": None,
        "derivatives": None,
        "different_license": None,
        "url": "https://spdx.org/licenses/Apache-2.0.html",
    }
    (version,) = detail.versions
    assert (version.id, version.name, version.base_model, version.created_at) == (
        "16x-ESRGAN", "Release", "ESRGAN 16x", "2019-07-06",
    )
    assert (version.trigger_words, version.images, version.early_access, version.nsfw) == ([], [], False, False)
    assert version.partial is False
    assert [file.model_dump() for file in version.files] == [
        {
            "id": "8701acac21c4c412a3e05242b389700c1248ff52314380793c045326f4c61dfb",
            "name": "16x-ESRGAN.pth",
            "size_bytes": 67249159,
            "sha256": "8701acac21c4c412a3e05242b389700c1248ff52314380793c045326f4c61dfb",
            "format": "PickleTensor",
            "precision": None,
            "file_type": "PyTorch",
            "primary": True,
            "download_url": "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/16x-ESRGAN.pth",
            "auth_provider_id": None,
        }
    ]


@pytest.mark.parametrize(
    "model_id, name, url",
    [
        (GITHUB_RELEASE, "1x-Archivist_AntiLines.pth",
         "https://github.com/Loganavter/Archivist-Project-Denoiser/releases/download/v1.0/1x-Archivist_AntiLines.pth"),
        (GITHUB_RAW, "4xNomos8kSC.pth", "https://github.com/Phhofm/models/raw/main/4xNomos8kSC/4xNomos8kSC.pth"),
        (RAW_GITHUB, "2xLiveActionV1_SPAN_490000.pth",
         "https://raw.githubusercontent.com/jcj83429/upscaling/f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.pth"),
        (HUGGING_FACE, "4x_foolhardy_Remacri.pth",
         "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth?download=true"),
        (MIXED, "4x-UltraSharpV2.safetensors",
         "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.safetensors?download=true"),
    ],
)
def test_the_primary_file_of_each_kind_of_direct_url(model_id, name, url):
    primary = _detail(model_id).versions[0].files[0]

    assert (primary.primary, primary.name, primary.download_url, primary.auth_provider_id) == (True, name, url, None)
    assert primary.id == primary.sha256 and len(primary.sha256) == 64


def test_the_first_direct_url_of_a_resource_wins_and_its_mirrors_are_skipped():
    raw = model(BACKBLAZE)
    oracle, backblaze, proton = raw["resources"][0]["urls"]

    assert _detail(BACKBLAZE).versions[0].files[0].download_url == oracle
    raw["resources"][0]["urls"] = [proton, backblaze, oracle]
    assert _detail(BACKBLAZE, _alone(BACKBLAZE, raw)).versions[0].files[0].download_url == backblaze
    assert other_downloads(raw) == []


def test_onnx_exports_are_never_files():
    three = _detail(THREE_FORMATS).versions[0].files
    onnx_first = _detail(ONNX_FIRST).versions[0].files

    assert [(file.format, file.file_type, file.primary) for file in three] == [
        ("PickleTensor", "PyTorch", True), ("SafeTensor", "PyTorch", False),
    ]
    assert [(file.name, file.primary) for file in onnx_first] == [("4xNomos8kSCHAT-S.pth", True)]
    assert _item(ONNX_FIRST).primary_sha256 == onnx_first[0].sha256
    for model_id in (ONNX_ONLY, ONNX_AND_DRIVE):
        assert (_item(model_id).mb_type, _item(model_id).primary_sha256, _detail(model_id).versions[0].files) == (
            None, None, [],
        )


@pytest.mark.parametrize("onnx", [{"platform": "onnx", "type": "pth"}, {"platform": "pytorch", "type": "onnx"}])
def test_a_resource_is_onnx_by_its_platform_or_its_type(onnx):
    raw = model(ORACLE)
    raw["resources"][0].update(onnx)

    assert _detail(ORACLE, _alone(ORACLE, raw)).versions[0].files == []
    assert other_downloads(raw) == ["ONNX"]


def test_the_first_pytorch_file_is_primary_and_listed_first():
    raw = model(THREE_FORMATS)
    raw["resources"][0]["platform"] = "other"

    files = _detail(THREE_FORMATS, _alone(THREE_FORMATS, raw)).versions[0].files

    assert [(file.format, file.file_type, file.primary) for file in files] == [
        ("SafeTensor", "PyTorch", True), ("PickleTensor", "other", False),
    ]


def test_file_ids_are_the_hashes_or_the_index_and_duplicates_go():
    raw = model("2x-GameUp-TSCUNet")
    first, fp16 = raw["resources"][:2]
    fp16["sha256"] = first["sha256"].upper()
    raw["resources"].append(
        {"platform": "pytorch", "type": "pth", "size": 1, "urls": ["https://github.com/o/r/releases/download/v1/extra.pth"]}
    )

    files = _detail("2x-GameUp-TSCUNet", _alone("2x-GameUp-TSCUNet", raw)).versions[0].files

    assert [file.id for file in files] == [first["sha256"], "4"]
    assert (files[1].sha256, files[1].name) == (None, "extra.pth")


def test_odd_resource_values_are_ignored():
    raw = model(ORACLE)
    resource = raw["resources"][0]
    resource.update(size=-1, sha256="not a hash", platform="tensorflow", type=None)
    raw["resources"].append("not a resource")

    (file,) = _detail(ORACLE, _alone(ORACLE, raw)).versions[0].files

    assert (file.size_bytes, file.sha256, file.file_type, file.format, file.id) == (None, None, "tensorflow", None, "0")
    assert file.name == "16x-ESRGAN.pth"


def test_a_resource_whose_url_lacks_its_extension_is_not_a_file():
    raw = model(ORACLE)
    raw["resources"][0]["type"] = "safetensors"

    assert _detail(ORACLE, _alone(ORACLE, raw)).versions[0].files == []


GRAYSCALE_THUMB = "https://openmodeldb.info/thumbs/c5b8354a01bb32171fab493f.jpg"
GRAYSCALE_SMALL = "https://openmodeldb.info/thumbs/small/fcbcdea19d949942313714fd.jpg"


def test_the_card_shows_the_models_own_thumbnail_resolved_on_openmodeldb():
    item = _item(GITHUB_RAW)

    assert (item.thumbnail.url, item.thumbnail.width, item.thumbnail.height) == (
        "https://openmodeldb.info/thumbs/79385d8f832b948a9ce1d339.jpg", 368, 296,
    )
    assert (item.thumbnail.nsfw, item.thumbnail.nsfw_level, item.thumbnail.kind) == (False, 1, "image")
    assert item.thumbnail.id and len(item.thumbnail.id) == 16
    assert _item(THREE_FORMATS).thumbnail.url == "https://openmodeldb.info/thumbs/32f56319a7a8f9fb411fc9f3.jpg"
    assert (_item(PAIR_WITHOUT_SIZES).thumbnail.url, _item(PAIR_WITHOUT_SIZES).thumbnail.width) == (
        "https://i.slow.pics/k1sDOhPk.webp", None,
    )
    assert _item("2x-GameUp-TSCUNet").thumbnail.url == "https://i.slow.pics/nrZhKdOx.png"


def test_without_its_own_thumbnail_the_card_shows_the_first_small_one():
    raw = model(GRAYSCALE)
    del raw["thumbnail"]
    standalone = model(STANDALONE_IMAGES)
    del standalone["thumbnail"], standalone["images"][0]["thumbnail"]

    assert _item(GRAYSCALE, _alone(GRAYSCALE, raw)).thumbnail.url == GRAYSCALE_SMALL
    assert _item(STANDALONE_IMAGES, _alone(STANDALONE_IMAGES, standalone)).thumbnail.url == (
        "https://openmodeldb.info" + standalone["images"][1]["thumbnail"]
    )


def test_a_model_thumbnail_on_a_dead_host_falls_back_to_the_small_one():
    raw = model(GRAYSCALE)
    raw["thumbnail"] = {"type": "paired", "LR": "https://imgsli.com/i/lr.jpg", "SR": "https://imgsli.com/i/sr.jpg"}

    assert _item(GRAYSCALE, _alone(GRAYSCALE, raw)).thumbnail.url == GRAYSCALE_SMALL


def test_without_any_thumbnail_the_card_has_no_image():
    raw = model(GRAYSCALE)
    del raw["thumbnail"], raw["images"][0]["thumbnail"]

    assert _item(GRAYSCALE, _alone(GRAYSCALE, raw)).thumbnail is None
    assert _item(ORACLE).thumbnail is None


def test_previews_start_with_the_models_thumbnail_then_the_gallery():
    standalone = _detail(STANDALONE_IMAGES).versions[0].images
    pairs = _detail(PAIR_WITHOUT_SIZES).versions[0].images
    grayscale = _detail(GRAYSCALE).versions[0].images

    assert [image.url for image in standalone] == [
        "https://openmodeldb.info/thumbs/d5ab7a8841cb834eb2e2a27d.jpg",
        "https://images2.imgbox.com/cf/bd/lq4EWYPp_o.png",
        "https://images2.imgbox.com/ba/81/OepEK0UZ_o.png",
    ]
    assert [image.url for image in pairs] == [
        "https://i.slow.pics/k1sDOhPk.webp", "https://i.slow.pics/BIRBcKK9.webp", "https://i.slow.pics/h9VzPczR.webp",
    ]
    assert [(image.url, image.width, image.height) for image in grayscale] == [
        (GRAYSCALE_THUMB, 366, 296), ("https://images2.imgbox.com/28/04/wus0QURa_o.png", None, None),
    ]
    assert model(GRAYSCALE)["images"][0]["LR"] not in [image.url for image in grayscale]
    assert len({image.id for image in standalone}) == 3


def test_images_on_dead_hosts_are_dropped():
    raw = model(GRAYSCALE)
    raw["images"] = [
        {"type": "paired", "LR": "https://imgsli.com/i/lr.jpg", "SR": "https://imgsli.com/i/sr.jpg"},
        {"type": "standalone", "url": "https://github.com/Phhofm/models/assets/14755670/2cf53d9b-a601-4fea-9440-84c968a23e50"},
        {"type": "standalone", "url": "https://private-user-images.githubusercontent.com/1/x.png?jwt=eyJ0eXAi"},
        {"type": "standalone", "url": "https://www.imgsli.com/i/y.jpg"},
        {"type": "standalone", "url": "https://i.ibb.co/x/kept.png"},
        {"type": "standalone", "url": "https://user-images.githubusercontent.com/1/kept.png"},
        {"type": "standalone", "url": "https://raw.githubusercontent.com/o/r/main/kept.png"},
    ]

    images = _detail(GRAYSCALE, _alone(GRAYSCALE, raw)).versions[0].images

    assert [image.url for image in images] == [
        GRAYSCALE_THUMB,
        "https://i.ibb.co/x/kept.png",
        "https://user-images.githubusercontent.com/1/kept.png",
        "https://raw.githubusercontent.com/o/r/main/kept.png",
    ]
    assert [image.url for image in _detail(GITHUB_RAW).versions[0].images] == [
        "https://openmodeldb.info/thumbs/79385d8f832b948a9ce1d339.jpg"
    ]


def test_a_model_with_nothing_left_has_no_preview_but_its_card_keeps_a_small_thumbnail():
    raw = model(GITHUB_RAW)
    del raw["thumbnail"]
    cat = _alone(GITHUB_RAW, raw)

    assert _detail(GITHUB_RAW, cat).versions[0].images == []
    assert _item(GITHUB_RAW, cat).thumbnail.url == "https://openmodeldb.info/thumbs/small/ab132cce7f26a2210ac16912.jpg"


def test_images_that_are_not_https_are_dropped_and_duplicates_listed_once():
    raw = model(ORACLE)
    good = "https://i.slow.pics/good.webp"
    raw["images"] = [
        {"type": "paired", "LR": "https://i.slow.pics/lr.webp", "SR": "http://i.slow.pics/plain.webp"},
        {"type": "paired", "SR": "javascript:alert(1)"},
        {"type": "paired", "SR": 'https://i.slow.pics/a".webp'},
        {"type": "standalone", "url": "https://user@i.slow.pics/x.webp"},
        {"type": "standalone", "url": good},
        {"type": "paired", "SR": good},
        "not an image",
        {"type": "standalone", "url": "/thumbs/own.jpg"},
    ]

    images = _detail(ORACLE, _alone(ORACLE, raw)).versions[0].images

    assert [image.url for image in images] == [good, "https://openmodeldb.info/thumbs/own.jpg"]


def test_the_previews_are_capped_thumbnail_included():
    raw = model(GRAYSCALE)
    raw["images"] = [{"type": "standalone", "url": f"https://i.slow.pics/{index}.webp"} for index in range(40)]

    images = _detail(GRAYSCALE, _alone(GRAYSCALE, raw)).versions[0].images

    assert len(images) == omdb.MAX_IMAGES == 12
    assert (images[0].url, images[-1].url) == (GRAYSCALE_THUMB, "https://i.slow.pics/10.webp")


@pytest.mark.parametrize(
    "value, summary, url",
    [
        ("CC-BY-NC-SA-4.0", "CC-BY-NC-SA-4.0", "https://spdx.org/licenses/CC-BY-NC-SA-4.0.html"),
        ("GPL-3.0-only", "GPL-3.0-only", "https://spdx.org/licenses/GPL-3.0-only.html"),
        (" WTFPL ", "WTFPL", "https://spdx.org/licenses/WTFPL.html"),
        ("MIT OR Apache-2.0", "MIT OR Apache-2.0", None),
        ("<b>x</b>", "<b>x</b>", None),
        (None, "No license stated", None),
        ("", "No license stated", None),
        (42, "No license stated", None),
    ],
)
def test_license(value, summary, url):
    found = license_from({"license": value})

    assert (found.summary, found.url) == (summary, url)


def test_a_model_without_a_license():
    assert _detail(NO_LICENSE).license.summary == "No license stated"


def test_the_description_starts_with_the_facts():
    detail = _detail(GITHUB_RAW)

    assert detail.description_html == (
        "<p>Architecture: ESRGAN · Scale: 4x · Input: image · Channels: 3 → 3 · Dataset: Nomos8k_sfw · "
        "Training: 85,000 iterations, 137 epochs</p>"
        "<p>4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr</p>"
    )


def test_facts_name_only_what_the_model_gives():
    cat = catalog()

    assert facts_line(cat.models[VIDEO], cat).startswith("Architecture: CAIN · Scale: 2x · Input: video")
    assert "Channels: 1 → 1" in facts_line(cat.models[GRAYSCALE], cat)
    assert facts_line(cat.models[GRAYSCALE], cat).endswith("Training: 23 epochs")
    assert "Training" not in facts_line(cat.models[MISSPELT_KEYS], cat)
    assert facts_line({}, cat) is None


def test_other_downloads_are_named_after_the_description():
    drive, mega, mixed = (_detail(model_id) for model_id in (GOOGLE_DRIVE_ONLY, MEGA_ONLY, MIXED))

    assert _text(drive).endswith("Other downloads on the OpenModelDB page: Google Drive")
    assert _text(mega).endswith("Other downloads on the OpenModelDB page: Mega")
    assert "Other downloads on the OpenModelDB page: ONNX" in _text(mixed)
    assert (
        '<p>Other downloads on the <a href="https://openmodeldb.info/models/3x-OmniSR-DF2K" '
        'rel="noopener noreferrer" target="_blank">OpenModelDB page</a>: Google Drive</p>'
    ) in drive.description_html
    assert "Other downloads" not in _detail(GITHUB_RAW).description_html
    assert other_downloads(model(ONNX_AND_DRIVE)) == ["ONNX", "Google Drive"]
    assert other_downloads(model(STANDALONE_IMAGES)) == ["Mega", "ONNX"]
    assert other_downloads(model(ONNX_FIRST)) == other_downloads(model(ONNX_ONLY)) == ["ONNX"]
    assert other_downloads(model(HTTP_ONLY)) == ["mufunyo.net"]
    assert other_downloads(model("1x-DeJpeg-Fatality-PlusULTRA")) == []


def test_other_downloads_are_named_once_in_order():
    raw = {
        "resources": [
            {"type": "pth", "urls": ["https://www.mediafire.com/file/a", "https://u.pcloud.link/x", "https://e1.pcloud.link/y"]},
            {"type": "onnx", "platform": "onnx", "urls": ["https://github.com/o/r/releases/download/v1/x.onnx"]},
            {"type": "pth", "urls": ["https://1drv.ms/u/s!x", "https://www.mediafire.com/file/b", "https://drive.proton.me/urls/z"]},
            {"type": "safetensors", "urls": ["https://icedrive.net/1/x", "https://www.dropbox.com/s/y", "https://WWW.Example.org/z"]},
            {"type": "onnx", "urls": ["https://mega.nz/file/z"]},
            {"type": "pth", "urls": ["not a url", ""]},
            {"type": "onnx", "urls": []},
        ]
    }

    assert other_downloads(raw) == [
        "MediaFire", "pCloud", "ONNX", "OneDrive", "Proton Drive", "Icedrive", "Dropbox", "example.org",
    ]


def test_an_esrgan_pth_model_has_no_compatibility_note():
    for model_id in (GITHUB_RAW, ORACLE, HUGGING_FACE, "1x-Debandurh-FS-Ultra-lite"):
        assert "Note: PotionUI" not in _detail(model_id).description_html, model_id


def test_a_model_of_another_architecture_gets_the_note_last():
    span = _detail(RAW_GITHUB)
    hat = _detail(ONNX_FIRST)

    assert span.description_html.endswith("<p>" + NOTE.format("SPAN ") + "</p>")
    assert _text(hat).endswith("Other downloads on the OpenModelDB page: ONNX " + NOTE.format("HAT "))


def test_an_esrgan_model_whose_primary_file_is_not_a_pth_gets_the_note():
    raw = model(GITHUB_RAW)
    raw["resources"][0].update(
        type="safetensors", urls=["https://github.com/Phhofm/models/raw/main/4xNomos8kSC/4xNomos8kSC.safetensors"]
    )

    detail = _detail(GITHUB_RAW, _alone(GITHUB_RAW, raw))

    assert detail.versions[0].files[0].format == "SafeTensor"
    assert _text(detail).endswith(NOTE.format("ESRGAN "))


def test_no_note_for_esrgan_plus_or_a_model_without_files():
    raw = model(GITHUB_RAW)
    raw["architecture"] = "esrgan+"

    assert "Note: PotionUI" not in _detail(GITHUB_RAW, _alone(GITHUB_RAW, raw)).description_html
    for model_id in (MEGA_ONLY, GOOGLE_DRIVE_ONLY, ONNX_ONLY, ONNX_AND_DRIVE):
        assert "Note: PotionUI" not in _detail(model_id).description_html, model_id


def test_the_note_without_a_known_architecture():
    raw = model(RAW_GITHUB)
    del raw["architecture"]
    cat = _alone(RAW_GITHUB, raw)

    assert compatibility_note(raw, cat, model_files(RAW_GITHUB, raw)) == NOTE.format("")
    assert compatibility_note(raw, cat, []) is None


def test_the_description_is_sanitised_markdown():
    raw = model(GITHUB_RAW)
    raw["description"] = (
        "# Title\n\nSee [the page](https://example.org/x) and <https://slow.pics/c/1>.\n\n"
        '<script>alert(1)</script><img src=x onerror="alert(1)"><a href="javascript:alert(1)">bad</a>'
    )

    markup = _detail(GITHUB_RAW, _alone(GITHUB_RAW, raw)).description_html

    assert "<h3>Title</h3>" in markup
    assert '<a href="https://example.org/x" rel="noopener noreferrer" target="_blank">the page</a>' in markup
    assert 'href="https://slow.pics/c/1"' in markup
    assert "script" not in markup and "onerror" not in markup and "javascript" not in markup and "<img" not in markup
    assert set(_tags_in(markup)) <= _SANITISED_TAGS


def test_inline_html_in_a_description_keeps_its_line_breaks():
    markup = _detail(INLINE_HTML).description_html

    assert "<br>" in markup and "✔ Color bleed" in markup


def test_without_markdown2_the_description_is_escaped_paragraphs(monkeypatch):
    monkeypatch.setitem(sys.modules, "markdown2", None)
    raw = model(GITHUB_RAW)
    raw["description"] = "one **two**\n\nthree <b>four</b>"

    markup = _detail(GITHUB_RAW, _alone(GITHUB_RAW, raw)).description_html

    assert markup.endswith("<p>one **two**</p><p>three &lt;b&gt;four&lt;/b&gt;</p>")


def test_a_very_long_description_is_not_converted(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "markdown2", type("M", (), {"markdown": staticmethod(lambda *a, **k: calls.append(1))}))
    text = "word " * (omdb.MAX_MARKDOWN_CHARS // 5 + 10)

    assert omdb.render_markdown(text).startswith("<p>word word")
    assert calls == []


def test_a_markdown2_failure_falls_back_to_paragraphs(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "markdown2", type("M", (), {"markdown": staticmethod(broken)}))

    assert omdb.render_markdown("a\n\nb <c>") == "<p>a</p><p>b &lt;c&gt;</p>"


def test_a_model_without_facts_or_description_has_no_description():
    cat = catalog()

    assert description_html("x", {}, cat) is None


def _text(detail):
    return html_to_text(detail.description_html, limit=100_000)


def _tags_in(markup):
    import re

    return [name.lower() for name in re.findall(r"</?([a-zA-Z0-9]+)", markup)]


async def test_the_first_load_reads_the_three_files_once_and_caches_them(clock):
    transport, cache = OpenModelDBTransport(), MemoryCache()
    source = make_source(transport, cache, clock)

    loaded = await source.catalog()
    again = await source.catalog()

    assert again is loaded and source.snapshot is loaded
    assert sorted(transport.calls) == sorted([MODELS, TAGS, ARCHITECTURES])
    assert len(loaded.models) == 27 and loaded.fetched_at == T0
    assert set(cache.ttls.values()) == {CATALOG_TTL_SECONDS}
    stored = [value for value in cache.entries.values() if isinstance(value.get("body"), dict) and ORACLE in value["body"]]
    assert stored and stored[0]["fetched_at"] == T0


async def test_concurrent_first_loads_share_one_download(clock):
    transport = OpenModelDBTransport()

    async def slow(url, params):
        await asyncio.sleep(0.05)
        return await transport(url, params)

    source = OpenModelDBSource(
        http_client=SourceHttp(fetch=slow, limits={"openmodeldb": (0.0, 2)}), cache_store=MemoryCache(), clock=clock
    )

    results = await asyncio.gather(source.catalog(), source.catalog(), source.search(SearchQuery()), source.info())

    assert results[0] is results[1]
    assert transport.count(MODELS) == 1
    assert len(results[2].items) == 24 and results[3].types


async def test_the_snapshot_is_kept_for_six_hours_then_downloaded_again(clock):
    transport = OpenModelDBTransport()
    source = make_source(transport, clock=clock)
    first = await source.catalog()

    clock.now += CATALOG_TTL_SECONDS - 1
    assert await source.catalog() is first
    clock.now += 1
    second = await source.catalog()

    assert second is not first and second.fetched_at == clock.now
    assert transport.count(MODELS) == 2


async def test_after_a_restart_a_fresh_cache_row_is_used_without_asking(clock):
    transport, cache = OpenModelDBTransport(), MemoryCache()
    await make_source(transport, cache, clock).catalog()
    clock.now += 3600

    restarted = make_source(transport, cache, clock)
    loaded = await restarted.catalog()

    assert transport.count(MODELS) == 1
    assert loaded.fetched_at == T0
    clock.now = T0 + CATALOG_TTL_SECONDS
    await restarted.catalog()
    assert transport.count(MODELS) == 2


async def test_the_real_cache_keeps_the_catalogue_six_hours(plugin_db, clock):
    create_tables()
    cache = Cache(clock=clock)
    transport = OpenModelDBTransport()
    await make_source(transport, cache, clock).catalog()

    clock.now += CATALOG_TTL_SECONDS - 60
    assert cache.purge() == 0
    assert len((await make_source(transport, cache, clock).catalog()).models) == 27
    assert transport.count(MODELS) == 1

    clock.now += 120
    await make_source(transport, cache, clock).catalog()
    assert transport.count(MODELS) == 2


async def test_a_cache_row_from_the_future_or_without_a_time_is_not_trusted(clock):
    transport, cache = OpenModelDBTransport(), MemoryCache()
    await make_source(transport, cache, clock).catalog()
    for key, value in cache.entries.items():
        cache.entries[key] = {"body": value["body"]}

    await make_source(transport, cache, clock).catalog()
    clock.now -= 3600
    await make_source(transport, cache, clock).catalog()

    assert transport.count(MODELS) == 3


async def test_a_failed_refresh_keeps_serving_the_previous_catalogue(clock, caplog):
    transport = OpenModelDBTransport()
    source = make_source(transport, clock=clock)
    first = await source.catalog()
    clock.now += CATALOG_TTL_SECONDS
    transport.responses[MODELS] = (503, {}, b"Service Unavailable")

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"):
        stale = await source.catalog()

    assert stale is first
    assert "could not refresh the catalogue" in caplog.text
    assert transport.count(MODELS) == 2
    clock.now += RETRY_SECONDS - 1
    assert _ids(await source.search(SearchQuery(limit=2))) == NEWEST[:2]
    assert transport.count(MODELS) == 2
    clock.now += 1
    transport.responses[MODELS] = "models"
    assert await source.catalog() is not first
    assert transport.count(MODELS) == 3


async def test_a_failed_first_load_is_raised_and_held_for_a_moment(clock):
    transport = OpenModelDBTransport()
    transport.responses[MODELS] = OSError("network is unreachable")
    source = make_source(transport, clock=clock)

    with pytest.raises(SourceUnavailable) as first:
        await source.catalog()
    with pytest.raises(SourceUnavailable) as held:
        await source.search(SearchQuery())

    assert "Could not reach OpenModelDB" in first.value.message
    assert held.value is not first.value and held.value.message == first.value.message
    assert transport.count(MODELS) == 1
    clock.now += FAILURE_HOLD_SECONDS
    transport.responses[MODELS] = "models"
    assert len((await source.catalog()).models) == 27
    assert transport.count(MODELS) == 2


async def test_a_catalogue_that_is_not_an_object_is_unavailable(clock):
    transport = OpenModelDBTransport()
    transport.responses[MODELS] = (200, {}, ["not", "an", "object"])

    with pytest.raises(SourceUnavailable):
        await make_source(transport, clock=clock).catalog()


async def test_a_challenge_page_is_unavailable(clock):
    transport = OpenModelDBTransport()
    transport.responses[MODELS] = (200, {"Content-Type": "text/html"}, b"<html>Just a moment...</html>")

    with pytest.raises(SourceUnavailable):
        await make_source(transport, clock=clock).catalog()


@pytest.mark.parametrize("failing", [TAGS, ARCHITECTURES])
async def test_tags_and_architectures_are_optional(clock, caplog, failing):
    transport = OpenModelDBTransport()
    transport.responses[failing] = (500, {}, b"oops")
    source = make_source(transport, clock=clock)

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"):
        item = (await source.search(SearchQuery(q=f"https://openmodeldb.info/models/{GITHUB_RAW}"))).items[0]
    tags = (await source.detail(GITHUB_RAW)).tags

    assert "showing the keys as labels" in caplog.text
    if failing == TAGS:
        assert (tags, item.type) == (["compression-removal", "jpeg", "photo", "restoration"], "ESRGAN")
    else:
        assert (tags[2], item.type, item.base_model) == ("Realistic", "esrgan", "esrgan 4x")


async def test_a_broken_cache_never_stops_the_catalogue(clock, caplog):
    class Broken:
        def get(self, *args, **kwargs):
            raise RuntimeError("no database")

        def set(self, *args, **kwargs):
            raise RuntimeError("no database")

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"):
        loaded = await make_source(OpenModelDBTransport(), Broken(), clock).catalog()

    assert len(loaded.models) == 27
    assert "Could not read the cache" in caplog.text and "Could not write the cache" in caplog.text


def test_entries_the_browser_cannot_use_are_left_out(caplog):
    body = models()
    body["../escape"] = body[ORACLE]
    body["bad id"] = body[ORACLE]
    body["4x-Not-An-Object"] = ["x"]

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"):
        cat = build_catalog(body)

    assert len(cat.models) == 27
    assert "left out 3 catalogue entries" in caplog.text


def test_the_sha256_index_prefers_a_model_that_can_be_downloaded():
    body = models()
    oracle_sha = body[ORACLE]["resources"][0]["sha256"]
    twin = dict(body[ORACLE], resources=[dict(body[ORACLE]["resources"][0], urls=["https://mega.nz/file/x"])])
    first = {"0-Copy-Not-Downloadable": twin, **body}

    cat = build_catalog(first)

    assert cat.by_sha256[oracle_sha][0] == ORACLE
    mega_sha = body[MEGA_ONLY]["resources"][0]["sha256"]
    assert cat.by_sha256[mega_sha][0] == MEGA_ONLY
    assert all(len(sha) == 64 and sha == sha.lower() for sha in cat.by_sha256)


async def test_find_sha256_ignores_case_and_answers_none_for_an_unknown_file(clock):
    source = make_source(OpenModelDBTransport(), clock=clock)
    sha = model(GITHUB_RAW)["resources"][0]["sha256"]

    found = await source.find_sha256(sha.upper())

    assert (found[0], found[2]["sha256"]) == (GITHUB_RAW, sha)
    assert found[1]["name"] == "Nomos8kSC" and found[3] is source.snapshot
    assert await source.find_sha256("0" * 64) is None


async def test_info_lists_the_architectures_and_scales_with_their_labels(clock):
    info = await make_source(OpenModelDBTransport(), clock=clock).info()

    assert info.model_dump() == {
        "id": "openmodeldb",
        "label": "OpenModelDB",
        "types": [
            {"value": "cain", "label": "CAIN"},
            {"value": "compact", "label": "Compact"},
            {"value": "dat", "label": "DAT"},
            {"value": "esrgan", "label": "ESRGAN"},
            {"value": "hat", "label": "HAT"},
            {"value": "omnisr", "label": "OmniSR"},
            {"value": "realplksr", "label": "RealPLKSR"},
            {"value": "span", "label": "SPAN"},
            {"value": "srformer", "label": "SRFormer"},
            {"value": "swift-srgan", "label": "Swift-SRGAN"},
            {"value": "swinir", "label": "SwinIR"},
            {"value": "tscunet", "label": "TSCUNet"},
        ],
        "base_models": [
            {"value": "1", "label": "1x"},
            {"value": "2", "label": "2x"},
            {"value": "3", "label": "3x"},
            {"value": "4", "label": "4x"},
            {"value": "8", "label": "8x"},
            {"value": "16", "label": "16x"},
        ],
        "sorts": [
            {"value": "downloadable", "label": "Downloadable first"},
            {"value": "newest", "label": "Newest"},
            {"value": "oldest", "label": "Oldest"},
            {"value": "name", "label": "Name"},
        ],
        "periods": [],
        "search_help": "Text, #tag (anime, photo, restoration, …), @author, or an openmodeldb.info model link",
        "types_label": "Architecture",
        "base_models_label": "Scale",
        "base_model_label": "Architecture",
        "notice": None,
    }


async def test_info_without_openmodeldb_still_lists_the_source(clock, caplog):
    transport = OpenModelDBTransport()
    transport.responses[MODELS] = (502, {}, b"Bad Gateway")

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"):
        info = await make_source(transport, clock=clock).info()

    assert (info.id, info.types, info.base_models, info.periods) == ("openmodeldb", [], [], [])
    assert (info.types_label, info.base_models_label, info.base_model_label) == ("Architecture", "Scale", "Architecture")
    assert [option.value for option in info.sorts] == ["downloadable", "newest", "oldest", "name"]
    assert "filter lists unavailable" in caplog.text


@pytest.fixture
def source(clock):
    return make_source(OpenModelDBTransport(), clock=clock)


async def test_the_default_search_lists_the_downloadable_models_first_by_pages_of_the_limit(source):
    first = await source.search(SearchQuery())
    second = await source.search(SearchQuery(cursor=first.next_cursor))

    assert (first.source, len(first.items), first.next_cursor) == ("openmodeldb", 24, "24")
    assert (len(second.items), second.next_cursor) == (3, None)
    items = first.items + second.items
    assert [found.mb_type for found in items] == ["upscaler"] * DOWNLOADABLE + [None] * (27 - DOWNLOADABLE)
    for group in (items[:DOWNLOADABLE], items[DOWNLOADABLE:]):
        dates = [omdb.date_of(models()[found.id]) for found in group]
        assert dates == sorted(dates, reverse=True)
    assert [found.id for found in items][:5] == NEWEST
    assert (items[DOWNLOADABLE - 1].id, items[-1].id) == (ORACLE, "1x-NMKD-YandereInpaint")
    assert len({found.id for found in items}) == 27


async def test_sorts(source):
    default = await source.search(SearchQuery(limit=50))
    downloadable = await source.search(SearchQuery(sort="downloadable", limit=50))
    newest = await source.search(SearchQuery(sort="newest", limit=50))
    oldest = await source.search(SearchQuery(sort="oldest", limit=3))
    by_name = await source.search(SearchQuery(sort="name", limit=50))

    assert _ids(default) == _ids(downloadable)
    dates = [omdb.date_of(models()[found.id]) for found in newest.items]
    assert dates == sorted(dates, reverse=True) and _ids(newest)[-1] == ORACLE
    assert _ids(newest).index(MISSPELT_KEYS) < _ids(newest).index(THREE_FORMATS)
    assert _ids(downloadable).index(THREE_FORMATS) < _ids(downloadable).index(MISSPELT_KEYS)
    assert _ids(oldest) == [ORACLE, NO_LICENSE, "1x-DeJpeg-Fatality-PlusULTRA"]
    names = [found.name.casefold() for found in by_name.items]
    assert names == sorted(names) and by_name.next_cursor is None


def test_equal_dates_are_ordered_by_name():
    body = {
        "b-Model": {"name": "beta", "date": "2024-01-01"},
        "a-Model": {"name": "Alpha", "date": "2024-01-01"},
        "c-Model": {"name": "gamma", "date": "2023-01-01"},
        "d-Model": {"name": "delta"},
    }
    body["c-Model"]["resources"] = [{"platform": "pytorch", "type": "pth", "urls": ["https://f002.backblazeb2.com/file/c.pth"]}]
    body["d-Model"]["resources"] = [{"platform": "pytorch", "type": "pth", "urls": ["https://f002.backblazeb2.com/file/d.pth"]}]
    body["a-Model"]["resources"] = [{"platform": "onnx", "type": "onnx", "urls": ["https://f002.backblazeb2.com/file/a.onnx"]}]

    cat = build_catalog(body)

    assert [entry.id for entry in cat.orders["newest"]] == ["a-Model", "b-Model", "c-Model", "d-Model"]
    assert [entry.id for entry in cat.orders["oldest"]] == ["c-Model", "a-Model", "b-Model", "d-Model"]
    assert [entry.id for entry in cat.orders["downloadable"]] == ["c-Model", "d-Model", "a-Model", "b-Model"]


async def test_an_unknown_sort_or_cursor_is_an_invalid_request(source):
    for query in (SearchQuery(sort="most_downloaded"), SearchQuery(cursor="abc"), SearchQuery(cursor="-1")):
        with pytest.raises(SourceInvalidRequest):
            await source.search(query)


async def test_a_cursor_past_the_end_is_an_empty_last_page(source):
    page = await source.search(SearchQuery(cursor="500"))

    assert (page.items, page.next_cursor) == ([], None)


async def test_the_limit_and_cursor_page_through(source):
    pages = []
    cursor = None
    while True:
        page = await source.search(SearchQuery(limit=10, cursor=cursor))
        pages.append(_ids(page))
        cursor = page.next_cursor
        if cursor is None:
            break

    assert [len(ids) for ids in pages] == [10, 10, 7]
    assert len({model_id for ids in pages for model_id in ids}) == 27


@pytest.mark.parametrize(
    "q, expected",
    [
        ("@kim2091", ["4x-UltraSharpV2", "2x-GameUp-TSCUNet", "1x-BroadcastToStudio-SPAN", "4x-UltraSharp"]),
        ("@ZARXRAX", ["2x-LD-Anime-Compact"]),
        ("@ skr ", ["2x-LD-Anime-Compact"]),
        ("@helaman", ["1x-DeNoise-realplksr-otf", "4x-Nomos8kSCHAT-S", "4x-Nomos8kSC", "4x-Nomos8kSCSRFormer"]),
        ("@kim", []),
        ("#Game Screenshots", ["2x-GameUp-TSCUNet"]),
        ("#game-screenshots", ["2x-GameUp-TSCUNet"]),
        ("#nope", []),
        ("nomos8k", ["4x-Nomos8kSCHAT-S", "4x-Nomos8kSC", "4x-Nomos8kSCSRFormer"]),
        ("UltraSharp DAT", ["4x-UltraSharpV2"]),
        ("SPAN", ["2x-LiveActionV1-SPAN", "4x-SPANkendata", "1x-BroadcastToStudio-SPAN"]),
        ("zzz", []),
    ],
)
async def test_the_search_box(source, q, expected):
    page = await source.search(SearchQuery(q=q, limit=50))

    assert (_ids(page), page.next_cursor) == (expected, None)


async def test_a_tag_matches_its_key_or_its_name(source):
    by_key = _ids(await source.search(SearchQuery(q="#photo", limit=50)))
    by_name = _ids(await source.search(SearchQuery(q="#Realistic", limit=50)))

    assert by_key == by_name and len(by_key) == 10
    assert GITHUB_RAW in by_key and ORACLE not in by_key


async def test_every_word_of_free_text_must_match(source):
    assert _ids(await source.search(SearchQuery(q="anime upscaler", limit=50))) == [
        "4x-UltraSharpV2", "4x-Nomos8kSCHAT-S", "4x-Nomos8kSCSRFormer", "2x-AnimeJaNai-Standard-v1-SuperUltraCompact",
    ]


@pytest.mark.parametrize(
    "q",
    [
        "https://openmodeldb.info/models/4x-Nomos8kSC",
        "openmodeldb.info/models/4x-Nomos8kSC",
        "https://www.openmodeldb.info/models/4x-nomos8ksc/?x=1#images",
        "http://openmodeldb.info/models/4x-Nomos8kSC",
    ],
)
async def test_a_pasted_link_is_that_model_whatever_the_filters(source, q):
    page = await source.search(SearchQuery(q=q, types=["dat"], base_models=["16"]))

    assert (_ids(page), page.next_cursor) == ([GITHUB_RAW], None)


async def test_a_link_to_an_unknown_model_or_past_its_page_is_empty(source):
    assert _ids(await source.search(SearchQuery(q="https://openmodeldb.info/models/4x-Nope"))) == []
    assert _ids(await source.search(SearchQuery(q="https://openmodeldb.info/models/4x-Nomos8kSC", cursor="24"))) == []


async def test_architecture_and_scale_filters(source):
    esrgan = await source.search(SearchQuery(types=["ESRGAN"], base_models=["4"], limit=50))
    two = await source.search(SearchQuery(types=["span", "compact"], sort="oldest", limit=50))
    scales = await source.search(SearchQuery(base_models=["16x", "8"], limit=50))

    assert _ids(esrgan) == ["4x-Nomos8kSC", "4x-Remacri", "4x-AnalogFrames-1-0", "4x-UltraSharp"]
    assert _ids(two) == [
        "2x-LD-Anime-Compact", "2x-AnimeJaNai-Standard-v1-SuperUltraCompact", "1x-BroadcastToStudio-SPAN",
        "4x-SPANkendata", "1x-BW-Denoise", "2x-LiveActionV1-SPAN", "1x-Book-Compact",
    ]
    assert _ids(scales) == [ONE_NAME_LIST, ORACLE]


async def test_a_known_architecture_without_models_is_an_empty_page(source):
    page = await source.search(SearchQuery(types=["esrgan+"]))

    assert (page.items, page.next_cursor) == ([], None)


@pytest.mark.parametrize("query", [SearchQuery(types=["not-an-arch"]), SearchQuery(base_models=["four"]), SearchQuery(base_models=["0"])])
async def test_unknown_filters_are_invalid_requests(source, query):
    with pytest.raises(SourceInvalidRequest):
        await source.search(query)


async def test_safe_max_level_and_period_change_nothing(source):
    plain = await source.search(SearchQuery(limit=5))
    safe = await source.search(SearchQuery(limit=5, safe=True, period="week"), max_level=1)

    assert safe == plain


async def test_each_search_hands_out_new_items(source):
    first = await source.search(SearchQuery(limit=1))
    first.items[0].installed = True
    first.items[0].base_models.append("changed")

    second = await source.search(SearchQuery(limit=1))

    assert second.items[0].installed is None and "changed" not in second.items[0].base_models


async def test_values_of_the_wrong_type_still_normalise(clock):
    transport = OpenModelDBTransport()
    body = models()
    body["4x-Odd"] = dict(body[ORACLE], date="2099-01-01", name={"not": "a name"}, author=[1, None, " x "], tags="anime")
    body["4x-Odd"]["resources"] = [{"type": "pth", "urls": ["https://f002.backblazeb2.com/o.pth", 7], "size": 1.0, "platform": ["x"]}]
    transport.responses[MODELS] = (200, {}, body)

    found = (await make_source(transport, clock=clock).search(SearchQuery(limit=1))).items[0]
    detail = await make_source(transport, clock=clock).detail("4x-Odd")
    (file,) = detail.versions[0].files

    assert (found.id, found.name, found.creator, detail.tags, found.mb_type) == ("4x-Odd", "4x-Odd", "x", [], "upscaler")
    assert (file.name, file.size_bytes, file.file_type) == ("o.pth", 1, None)


async def test_a_model_that_cannot_be_normalised_is_skipped(source, monkeypatch, caplog):
    real = omdb.normalize_item

    def failing(model_id, raw, catalog):
        if model_id == NEWEST[0]:
            raise ValueError("unexpected shape")
        return real(model_id, raw, catalog)

    monkeypatch.setattr(omdb, "normalize_item", failing)

    with caplog.at_level(logging.WARNING, logger="plugins.model-browser"):
        page = await source.search(SearchQuery(limit=3))

    assert _ids(page) == NEWEST[1:3] and page.next_cursor == "3"
    assert "Skipping an OpenModelDB model" in caplog.text


async def test_detail_by_id_link_or_another_case(source):
    plain = await source.detail(GITHUB_RAW)
    folded = await source.detail("4X-NOMOS8KSC")
    linked = await source.detail(f"https://openmodeldb.info/models/{GITHUB_RAW}")

    assert plain == folded == linked
    assert (plain.id, plain.mb_type, plain.versions[0].files[0].name) == (GITHUB_RAW, "upscaler", "4xNomos8kSC.pth")


async def test_detail_of_a_model_that_cannot_be_downloaded(source):
    detail = await source.detail(GOOGLE_DRIVE_ONLY)

    assert (detail.mb_type, detail.versions[0].files) == (None, [])
    assert detail.versions[0].id == GOOGLE_DRIVE_ONLY


async def test_detail_of_an_unknown_model_is_not_found(source):
    with pytest.raises(SourceNotFound):
        await source.detail("4x-Does-Not-Exist")


@pytest.mark.parametrize("model_id", ["../etc/passwd", "org/repo", "", "a" * 201, "4x x"])
async def test_detail_of_a_malformed_id_is_not_found_without_asking(clock, model_id):
    transport = OpenModelDBTransport()

    with pytest.raises(SourceNotFound):
        await make_source(transport, clock=clock).detail(model_id)

    assert transport.calls == []


async def test_detail_when_openmodeldb_is_down_is_unavailable(clock):
    transport = OpenModelDBTransport()
    transport.responses[MODELS] = (503, {}, b"")

    with pytest.raises(SourceUnavailable):
        await make_source(transport, clock=clock).detail(GITHUB_RAW)


@pytest.mark.parametrize(
    "q, expected",
    [
        ("", ParsedQuery()),
        ("   ", ParsedQuery()),
        ("@", ParsedQuery()),
        ("#", ParsedQuery()),
        ("@kim2091", ParsedQuery(kind="author", value="kim2091")),
        ("#Game  Screenshots", ParsedQuery(kind="tag", value="Game Screenshots")),
        ("UltraSharp  V2", ParsedQuery(kind="text", value="UltraSharp  V2", words=("ultrasharp", "v2"))),
        ("https://openmodeldb.info/models/4x-UltraSharp", ParsedQuery(kind="model", value="4x-UltraSharp")),
    ],
)
def test_parse_query(q, expected):
    assert parse_query(q) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("https://openmodeldb.info/models/4x-UltraSharp", "4x-UltraSharp"),
        ("openmodeldb.info/models/4x-UltraSharp/", "4x-UltraSharp"),
        ("https://openmodeldb.info/models/4x-UltraSharp?compare=1", "4x-UltraSharp"),
        ("https://openmodeldb.info/", None),
        ("https://openmodeldb.info/models/", None),
        ("https://openmodeldb.info/architectures/esrgan", None),
        ("https://openmodeldb.info/models/..%2Fx", None),
        ("https://evil.example/models/4x-UltraSharp", None),
        ("ftp://openmodeldb.info/models/4x-UltraSharp", None),
        ("https://openmodeldb.info/models/4x UltraSharp", None),
        ("4x-UltraSharp", None),
    ],
)
def test_parse_model_url(text, expected):
    assert parse_model_url(text) == expected
