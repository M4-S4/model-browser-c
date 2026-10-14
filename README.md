# Model Browser C

Browse models on CivitAI (civitai.com and civitai.red), Hugging Face, OpenModelDB and
CivArchive from inside PotionUI, and download the file you pick into the right model folder
through PotionUI's own download queue. The model then shows up on the Models page with its
description, previews, tags, base model and license filled in. Admins and users can both use
it: an admin decides per user group who may browse, request or download, and what users see
follows the instance's NSFW content policy.

## Getting started

This is **Model Browser C**, the Model Browser with the CivArchive source. Use it instead of
the Model Browser of the PotionUI marketplace, not next to it: both keep their data in the
same tables and declare the same model fields, notifications and plan limits, so PotionUI
refuses to enable the second one. Switching from one to the other keeps group rights,
requests and plan limits; plugin settings are per plugin and have to be set again.

The plugin needs PotionUI 0.0.16 or newer. Install it by placing this folder in
`content/plugins/local/model-browser-c`, then Admin → Plugins → Scan for Plugins, and
enable it.

Once the plugin is enabled, a **Model Browser** entry appears in the sidebar; it opens the
page `/plugins/model-browser-c`. The plugin keeps its own database tables (`model_browser_*`,
created when the application starts). It is told by PotionUI when one of its downloads has
landed, and a background task sweeps once a minute for the rest (failures, cancellations,
downloads finished while the plugin was disabled); disabling the plugin stops that task.

Browsing needs no credential. Downloads that need a login are authenticated by the CivitAI
and Hugging Face provider plugins, which hold the API key and the token. Admin → Plugins
shows two setup steps for this plugin, each marked done once that provider is initialised:

- Configure the CivitAI provider (API key) so logged-in downloads work.
- Configure the Hugging Face provider (token) for gated repositories.

Without them, models that require a login fail with 401.

## Settings

In Admin → Plugins → Model Browser → Settings:

- **NSFW in the browser** (default: follow the content policy): "Blur at most" or "Never
  show". It can only make the instance and group content policy stricter.
- **Highest CivitAI image level treated as safe** (PG, PG-13, R or X; default PG-13): images
  above this level count as NSFW and follow the content policy; models whose images are all
  above it count as NSFW models.
- **Hide CivitAI models depicting real people** (Show / Hide; Show by default).
- **CivitAI API base URL** (default `https://civitai.com/api/v1`): switch to
  `https://civitai.red/api/v1` if civitai.com starts filtering the API to SFW content.
- **Enable CivArchive (experimental, unofficial)** (Off / On; Off by default); read the CivArchive
  section below before turning it on.
- **Search cache TTL (minutes)** (default 10, from 1 to 1440): raw answers from the sources
  are cached in the database and filtered per user on every request.

## Permissions

Admins edit the rights of each user group in Admin → Plugins → Model Browser → Permissions.
There are four rights:

- **browse**: search the sources and view models.
- **request**: ask for a file the user may not download; an approver decides.
- **download**: download a file into PotionUI directly.
- **approve**: approve or reject other users' requests; holders of this right are notified
  of new requests.

Each group also has parameters that bound its downloads and requests: the sources, the model
types (none listed means every type), the largest file size in GB (none means no limit), who
the model is assigned to (the user, or one of their groups picked when downloading), and, for
downloads only, whether a remote destination is allowed.

A user's rights are the merge of their groups:

- a right is granted when at least one of the user's groups grants it;
- the download parameters come only from the groups that grant `download`, the request
  parameters only from the groups that grant `request`;
- sources and model types are combined (a group that lists no type allows every type); the
  largest size limit wins, and no limit beats any limit; the model is assigned to a group as
  soon as one group says so; a remote destination is allowed when one group allows it.

Administrators may do everything, with no parameter limiting them.

Defaults, for a group whose rights were never edited: all users may browse and request,
admins have every right, other groups have none. The default sources are CivitAI, Hugging
Face and OpenModelDB: CivArchive has to be added to a group's sources explicitly.

## Browsing and downloading

- Search box with `#tag`, `@creator`, `$#id1,id2` and pasted model links (the syntax
  varies per source, see below), filters (types, base model, period), sort, infinite
  scroll; badges for NSFW, installed, early access, login-gated and non-downloadable
  models. The search state lives in the URL.
- Detail drawer with versions, previews, trigger words, files, licence and the source link;
  download dialog with model type, subfolder, file name, remote destination and target
  group (when allowed); downloads tray with live status.
- A download goes through PotionUI's own download queue, into the folder of the chosen model
  type. When it is done, the model is assigned to the user (or to the chosen group) and its
  description, safe previews, trigger words, tags, **Base model** and **License** fields are
  filled in; the model's "Mirrors" section links back to the source. A file that is already
  on disk is only assigned, not downloaded again.

## Requests and approvals

- A file row shows **Request** to a user who may request but not download it; the dialog
  takes the model type and a note for the approvers.
- The **Requests** drawer (toolbar button, with the count of requests to review) lists the
  user's own requests and, for approvers, every open request with **Approve** (model type
  override, note) and **Reject** (note). The author can cancel an open request.
- Approving re-checks the requester's own rights and NSFW settings, then queues the file for
  them (or only assigns the model when it is already installed). Other approved requesters
  of the same file get it when it arrives.

## NSFW content

What each user sees follows the instance's content policy, the plugin's NSFW settings above
and the user's own sensitive content setting.

- NSFW previews reach the page as a blurred placeholder only, and are revealed on click when
  the content policy allows.
- The **Sensitive content** control on the page changes the user's app-wide sensitive
  content setting, within the limits of the policy. With it set to Hide, NSFW models are
  hidden from the results.
- Hugging Face repositories are NSFW when tagged `not-for-all-audiences`; the CivitAI image
  level setting does not apply to them.

## Quotas

Admin → Plans offers two limits for this plugin: **Model downloads per month** and
**Downloaded model bytes per month**. They count the files queued from the Model Browser for
a user, by direct download or by an approved request (an approval counts for the requester).
The plugin checks them before queueing a file and refuses the download or the approval when
the limit is reached; a failed download is refunded. Admins are exempt when the plans say so.

## Notifications

Each user can tune these in their notification preferences:

- **Model Browser: new model request**: sent to approvers when someone asks for a model.
- **Model Browser: request approved** / **Model Browser: request declined**: sent to the
  requester.
- **Model Browser: model ready** / **Model Browser: download failed**: sent to the user the
  download was for.

The request notifications link to the page's Requests drawer.

## Sources

### CivitAI

- civitai.com and civitai.red serve the same catalogue through the same API. Model pages
  link to civitai.red for an NSFW model and to civitai.com otherwise; pasted links from
  either host are understood.
- With the **CivitAI API base URL** set to the civitai.red host, download links are kept on
  civitai.com, so the CivitAI provider still authenticates them.
- Search: text, `#tag`, `@creator`, `$#id1,id2` or a civitai.com / civitai.red model link.
  Early access and login-only files are marked.

### Hugging Face

- Browsing shows what is public: no token is sent while searching. Gated repositories are
  listed and marked "Needs login"; their downloads use the Hugging Face provider's token.
- Search: `org/repo`, `@author`, `#tag`, free text or a pasted huggingface.co link. Sorts:
  trending, downloads, likes, updated, newest; there is no period filter.
- A repository has a single version, `main`.

### OpenModelDB (upscalers)

- The whole catalogue is cached for six hours and searched by the plugin: text, `#tag`,
  `@author`, a pasted openmodeldb.info model link, architecture and scale filters,
  "Downloadable first" by default, then date or name.
- Only PyTorch files on a direct host are offered (Hugging Face, GitHub releases or raw
  files, Oracle object storage, Backblaze; no ONNX, no archives). A model that only lives on
  Google Drive, Mega or the like shows "Not downloadable" and the link to its OpenModelDB
  page.
- Downloads are filed as upscalers. PotionUI's built-in upscaler runs ESRGAN `.pth` models
  only, so other architectures carry a note in their description.
- An upscaler already on disk also gets its name, description, previews and licence from
  the catalogue.

### CivArchive (experimental, opt-in)

CivArchive is off by default; turn on **Enable CivArchive (experimental, unofficial)** to
use it. Before you do, know that:

- civarchive.com is an unofficial community index of CivitAI, not an official API. The page
  shows this notice to users: deleted models may be unavailable, or come from third-party
  mirrors.
- Models deleted from CivitAI are served from their Hugging Face mirrors when they have one;
  torrent-only models show "Not downloadable".
- The plugin sends at most one request per second to the site.
- Turning the setting on does not let users other than admins download from it: CivArchive
  is not in the default sources, so add it to the sources of each group that should use it.

Search: text, `#tag`, `@user`, a civarchive.com / civitai.com model link, CivitAI type and
base model filters, sort and period. A model's other versions load when their tab is opened.
While the setting is on, models already on disk can also be identified through CivArchive.

## Troubleshooting

- **A download fails with 401**: the model needs a login. Configure the CivitAI provider
  (API key) or the Hugging Face provider (token); the setup steps in Admin → Plugins show
  whether each one is ready.
- **A user sees Request instead of Download, or cannot request a file**: check their groups in
  the Permissions tab: the right itself, then the sources, model types and size limit of the
  groups that grant it.
- **A download or an approval is refused for a limit**: the user (for an approval, the
  requester) has reached a monthly limit set in Admin → Plans.
- **A model shows "Not downloadable"**: PotionUI has no model type for this kind of file, or
  the source offers no file the plugin can fetch (see OpenModelDB and CivArchive above).
- **NSFW models are missing**: the user's sensitive content setting is Hide, or the content
  policy or the **NSFW in the browser** setting does not allow them.
- **CivitAI only returns SFW results**: set the **CivitAI API base URL** to
  `https://civitai.red/api/v1`.
- **CivArchive is missing from the sources**: turn on **Enable CivArchive (experimental,
  unofficial)**.
- **Checking that the plugin is up**: `GET /api/plugins/model-browser-c/health` answers with
  its status and version to any signed-in user. The plugin's lines in the server log start
  with `[MODEL-BROWSER]`.
