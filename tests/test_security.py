import re

import pytest

from backend.security import DOWNLOAD_HOSTS, html_to_text, is_allowed_download_url, sanitize_html


@pytest.mark.parametrize(
    "source, url",
    [
        ("civitai", "https://civitai.com/api/download/models/135867"),
        ("civitai", "https://civitai.com/api/download/models/135867?fileId=99264"),
        ("civitai", "https://CIVITAI.com:443/api/download/models/1"),
        ("huggingface", "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"),
        ("huggingface", "https://cdn-lfs.huggingface.co/repos/ab/cd/file"),
        ("openmodeldb", "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth"),
        ("openmodeldb", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
        ("openmodeldb", "https://objects.githubusercontent.com/github-production-release-asset/1"),
        ("openmodeldb", "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax/b/models/o/4x.pth"),
        ("openmodeldb", "https://raw.githubusercontent.com/terrainer/AI-Upscaling-Models/main/4xSPANkendata/x.onnx"),
        ("openmodeldb", "https://f002.backblazeb2.com/file/ESRGAN/_RELEASE/4x_Fatality_01_375000_G.pth"),
        ("civarchive", "https://civitai.com/api/download/models/1"),
        ("civarchive", "https://civitai.com/api/download/models/181422?type=Training%20Data"),
        ("civarchive", "https://huggingface.co/baycon/for_Colab/resolve/main/lowra_v10.safetensors"),
        ("civarchive", "https://huggingface.co/tim9292654/lora/resolve/main/%E7%BB%86%E5%8C%96/x.safetensors"),
    ],
)
def test_allowed_download_urls(source, url):
    assert is_allowed_download_url(source, url)


@pytest.mark.parametrize(
    "source, url",
    [
        ("civitai", "http://civitai.com/api/download/models/1"),
        ("civitai", "ftp://civitai.com/api/download/models/1"),
        ("civitai", "https://civitai.com:8443/api/download/models/1"),
        ("civitai", "//civitai.com/api/download/models/1"),
        ("civitai", "civitai.com/api/download/models/1"),
        ("civitai", "https://civitai.red/api/download/models/1"),
        ("civitai", "https://www.civitai.com/api/download/models/1"),
        ("civitai", "https://civitai.com.evil.example/api/download/models/1"),
        ("civitai", "https://evilcivitai.com/api/download/models/1"),
        ("civitai", "https://huggingface.co/x/y/resolve/main/z"),
        ("huggingface", "https://evilhuggingface.co/x"),
        ("huggingface", "https://huggingface.co.evil.example/x"),
        ("openmodeldb", "https://drive.google.com/file/d/abc/view"),
        ("openmodeldb", "https://mega.nz/file/abc"),
        ("openmodeldb", "https://cdn-lfs.huggingface.co/x"),
        ("openmodeldb", "https://oraclecloud.com/x"),
        ("openmodeldb", "https://backblazeb2.com/x"),
        ("openmodeldb", "https://githubusercontent.com/x"),
        ("openmodeldb", "https://user-images.githubusercontent.com/x.png"),
        ("openmodeldb", "https://www.dropbox.com/s/x/model.pth"),
        ("openmodeldb", "https://u.pcloud.link/publink/show?code=x"),
        ("civarchive", "https://civarchive.com/files/model.safetensors"),
        ("civarchive", "https://www.civarchive.com/x"),
        ("civarchive", "https://civarchive.com/api/download/models/1"),
        ("civarchive", "https://civitai.red/api/download/models/1"),
        ("civarchive", "https://cdn-lfs.huggingface.co/repos/ab/cd/file"),
        ("civarchive", "https://civitasbay.org/torrents/3A5909181098853299CEA64112B70284D199B24C"),
        ("civarchive", "https://tensordrive.app/files/fc6defee-4bb2-432f-9a4d-7f6fbeb9daef"),
        ("civarchive", "https://huggingface.co/tim9292654/lora/resolve/main/细化/x.safetensors"),
        ("civarchive", "https://huggingface.co/QEEWDFHGFH/052723/resolve/main/x (1).safetensors"),
        ("civitai", "https://127.0.0.1/api/download/models/1"),
        ("civitai", "https://10.0.0.5/api/download/models/1"),
        ("civitai", "https://[::1]/api/download/models/1"),
        ("civitai", "https://[fd00::1]/x"),
        ("civitai", "https://2130706433/x"),
        ("civitai", "https://127.1/x"),
        ("civitai", "https://localhost/x"),
        ("huggingface", "https://huggingface.localhost/x"),
        ("civitai", "https://user:pass@civitai.com/api/download/models/1"),
        ("civitai", "https://civitai.com@evil.example/x"),
        ("civitai", "https://evil.example\\@civitai.com/x"),
        ("civitai", "https://civitai.com/api/download/models/1 x"),
        ("civitai", "https://civitai.com/api/\ndownload"),
        ("civitai", "https://civitai.com./api/download/models/1"),
        ("civitai", "https://civitаi.com/api/download/models/1"),
        ("civitai", "https://civit%61i.com/api/download/models/1"),
        ("unknown", "https://civitai.com/api/download/models/1"),
        ("civitai", None),
        ("civitai", ""),
        ("civitai", "https://"),
        ("civitai", "https://[broken/x"),
    ],
)
def test_refused_download_urls(source, url):
    assert not is_allowed_download_url(source, url)


def test_every_source_has_an_allowlist():
    assert set(DOWNLOAD_HOSTS) == {"civitai", "huggingface", "openmodeldb", "civarchive"}


@pytest.mark.parametrize(
    "repo, path",
    [
        ("latent-consistency/lcm-lora-sdxl", "pytorch_lora_weights.safetensors"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors"),
        ("stabilityai/stable-diffusion-xl-base-1.0", "text_encoder/model.onnx"),
        ("org/demo", "sub dir/my model+v2 (final).safetensors"),
        ("org/demo", "q/model-Q4_K_M.gguf"),
    ],
)
def test_hugging_face_resolve_urls_are_allowed(repo, path):
    from backend.sources.huggingface import resolve_url

    url = resolve_url(repo, path)

    assert url.startswith(f"https://huggingface.co/{repo}/resolve/main/")
    assert is_allowed_download_url("huggingface", url)
    assert not is_allowed_download_url("civitai", url)


_KEPT_TAGS = {"p", "br", "b", "strong", "i", "em", "u", "s", "ul", "ol", "li", "h3", "h4", "blockquote", "code", "pre", "hr", "a"}


def _tags(output):
    return re.findall(r"<\s*/?\s*([a-zA-Z0-9]+)", output)


def _assert_safe(output):
    assert set(_tags(output)) <= _KEPT_TAGS, output
    for attributes in re.findall(r"<[a-zA-Z0-9]+(\s[^>]*)>", output):
        names = set(re.findall(r'\s([a-zA-Z-]+)="', attributes))
        assert names <= {"href", "rel", "target"}, output
    lowered = output.lower()
    for needle in ("<script", "<img", "<iframe", "<svg", "<style", "javascript:", "onerror=", "onload=", "onclick="):
        assert needle not in lowered, output


def test_formatting_is_kept():
    source = (
        "<p>One <strong>bold</strong> and <em>italic</em>, <b>b</b> <i>i</i> <u>u</u> <s>s</s></p>"
        "<ul><li>a</li><li>b</li></ul><ol><li>c</li></ol><blockquote>quote</blockquote>"
        "<pre><code>x = 1</code></pre><hr><p>line<br>break</p>"
    )
    output = sanitize_html(source)

    assert output == (
        "<p>One <strong>bold</strong> and <em>italic</em>, <b>b</b> <i>i</i> <u>u</u> <s>s</s></p>"
        "<ul><li>a</li><li>b</li></ul><ol><li>c</li></ol><blockquote>quote</blockquote>"
        "<pre><code>x = 1</code></pre><hr><p>line<br>break</p>"
    )


def test_headings_shrink_to_h3_and_h4():
    output = sanitize_html("<h1>A</h1><h2>B</h2><h3>C</h3><h6>D</h6>")

    assert output == "<h3>A</h3><h3>B</h3><h4>C</h4><h4>D</h4>"


def test_links_keep_only_an_http_href_and_open_in_a_new_tab():
    output = sanitize_html(
        '<a target="_self" rel="ugc" href="https://huggingface.co/x" onclick="steal()" style="color:red">HF</a>'
    )

    assert output == '<a href="https://huggingface.co/x" rel="noopener noreferrer" target="_blank">HF</a>'


@pytest.mark.parametrize(
    "href",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        " javascript:alert(1)",
        "jav&#x61;script:alert(1)",
        "java\tscript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "vbscript:msgbox(1)",
        "//evil.example/x",
        "/relative/path",
        "mailto:someone@example.com",
        "",
    ],
)
def test_a_link_without_a_safe_href_keeps_its_text_only(href):
    output = sanitize_html(f'<p><a href="{href}">click</a> after</p>')

    assert output == "<p>click after</p>"
    _assert_safe(output)


def test_scripts_styles_frames_and_media_are_dropped_with_their_content():
    output = sanitize_html(
        "<p>before</p><script>alert('x')</script><style>p{color:red}</style>"
        "<iframe src='https://evil.example'>frame text</iframe>"
        "<video src='x.mp4'><source src='y.mp4'>no video</video>"
        "<svg><script>alert(2)</script><text>svg</text></svg><p>after</p>"
    )

    assert output == "<p>before</p><p>after</p>"


def test_images_and_event_handlers_are_dropped():
    output = sanitize_html(
        '<p onclick="x()">text<img src="x.png" onerror="alert(1)">more</p>'
        '<img src=x onerror=alert(1)//><p style="background:url(javascript:alert(1))">styled</p>'
    )

    assert output == "<p>textmore</p><p>styled</p>"
    _assert_safe(output)


def test_unknown_tags_are_unwrapped_and_blocks_become_line_breaks():
    output = sanitize_html(
        '<div><span style="color:rgb(250, 176, 5)">Update</span></div><div>next</div>'
        "<table><tr><td>a</td><td>b</td></tr></table>"
    )

    assert output == "Update<br>next<br>a b <br>"
    _assert_safe(output)


def test_text_is_escaped_and_entities_survive():
    output = sanitize_html("<p>&lt;script&gt;alert(1)&lt;/script&gt; Tom &amp; Jerry 1 < 2 > 0 \"quoted\"</p>")

    assert output == '<p>&lt;script&gt;alert(1)&lt;/script&gt; Tom &amp; Jerry 1 &lt; 2 &gt; 0 "quoted"</p>'
    _assert_safe(output)


def test_nested_and_unbalanced_tags_are_closed():
    output = sanitize_html("<p><strong><em>deep</strong> tail")

    assert output == "<p><strong><em>deep</em></strong> tail</p>"


def test_stray_end_tags_are_ignored():
    assert sanitize_html("</p>text</strong></a>") == "text"


def test_attribute_values_are_escaped():
    output = sanitize_html('<a href="https://x.example/?a=1&b=&quot;2&quot;">x</a>')

    assert output == '<a href="https://x.example/?a=1&amp;b=&quot;2&quot;" rel="noopener noreferrer" target="_blank">x</a>'


@pytest.mark.parametrize(
    "payload",
    [
        "<scr<script>ipt>alert(1)</script>",
        "<<script>script>alert(1)<</script>/script>",
        "<img src=x:alert(1) onerror=eval(src)>",
        "<svg/onload=alert(1)>",
        "<a href=javascript&colon;alert(1)>x</a>",
        "<math><mtext><table><mglyph><style><img src=x onerror=alert(1)>",
        "<iframe srcdoc='&lt;script&gt;alert(1)&lt;/script&gt;'></iframe>",
        "<!--<img src=x onerror=alert(1)>-->",
        "<p title='</p><img src=x onerror=alert(1)>'>t</p>",
        '<a href="https://ok.example" onmouseover="alert(1)">ok</a>',
        "<object data='javascript:alert(1)'></object><embed src='javascript:alert(1)'>",
        "<form action='javascript:alert(1)'><button>go</button></form>",
        "<details open ontoggle=alert(1)>",
        "<body onload=alert(1)>",
        "<p>unclosed <b>bold <i>italic",
    ],
)
def test_known_injection_payloads_come_out_inert(payload):
    _assert_safe(sanitize_html(payload))


def test_the_real_civitai_description_is_kept_readable():
    source = (
        '<p>HF: <a target="_blank" rel="ugc" href="https://huggingface.co/OedoSoldier/detail-tweaker-lora">'
        "https://huggingface.co/OedoSoldier/detail-tweaker-lora</a></p>"
        "<p><strong>Note: use a negative weight to reduce details!</strong></p>"
    )

    assert sanitize_html(source) == (
        '<p>HF: <a href="https://huggingface.co/OedoSoldier/detail-tweaker-lora" rel="noopener noreferrer" '
        'target="_blank">https://huggingface.co/OedoSoldier/detail-tweaker-lora</a></p>'
        "<p><strong>Note: use a negative weight to reduce details!</strong></p>"
    )


def test_nothing_gives_an_empty_string():
    assert sanitize_html(None) == ""
    assert sanitize_html("") == ""
    assert sanitize_html("<script>only</script>") == ""


def test_html_to_text_flattens_and_drops_scripts():
    text = html_to_text("<h1>Title</h1><p>One &amp; two</p><script>alert(1)</script><ul><li>a</li><li>b</li></ul>")

    assert text == "Title One & two a b"


def test_html_to_text_cuts_at_a_word_with_an_ellipsis():
    text = html_to_text("<p>The quick brown fox jumps over the lazy dog</p>", limit=20)

    assert text == "The quick brown…"
    assert len(text) <= 20


def test_html_to_text_short_text_is_untouched():
    assert html_to_text("<p>short</p>", limit=20) == "short"
    assert html_to_text(None) == ""
    assert html_to_text("<p>text</p>", limit=0) == ""
