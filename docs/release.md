# Releasing

1. Bump `version` in the metadata block of `shape_arrange.py` and add a
   section for it to `CHANGELOG.md`. Commit and push.
2. Publish a GitHub release whose tag matches the version:
   `gh release create v0.9.0 --title "Shape-Aware Arrange v0.9.0" --notes ""`
   (or **Draft a new release** in the GitHub UI).
3. The Release workflow tests the tagged code, checks the tag against the
   plugin version, attaches `shape_arrange_any.py` (Orca Cloud requires an
   OS/arch suffix in the file name; `any` means pure Python), fills empty
   release notes from `CHANGELOG.md`, and publishes the version to Orca
   Cloud with those notes as the changelog.

Orca Cloud publishing needs no secrets. The repository is connected to the
plugin under **Edit plugin > GitHub publishing** on Orca Cloud, and the
workflow proves where it runs with a short-lived OIDC token from GitHub.
Orca Cloud only accepts runs triggered by a published release, so pushing
a tag on its own does nothing. If a run fails, fix the cause and re-run it.

## Orca Cloud listing

The listing text and images are uploaded by hand on Orca Cloud (**Edit plugin**). The release workflow only
publishes the plugin file and the changelog (cut to the 4000 characters Orca Cloud keeps).

The listing is English only. The description is the one block in
[orca-cloud-description.md](orca-cloud-description.md): under 4000 characters, it links to README.md for the full
description and already ends with the other five plugins, so nothing else is pasted after it.

All images are PNG and under 2 MB.

| Listing slot | File | Size |
|---|---|---|
| Plugin icon / store tile | `assets/orca-cloud/orca_organizer-tile.png` | 512x512 |
| Main (cover) image | `assets/orca-cloud/orca_organizer-main-en.png` | 1600x900 |
| Square thumbnail, where the listing asks for one | `assets/orca-cloud/orca_organizer-main-square-en.png` | 900x900 |
| Gallery 1, "Fit shapes closer together" | `assets/orca-cloud/orca_organizer-feature-01-shape-en.png` | 1600x900 |
| Gallery 2, "Fill the plate with copies" | `assets/orca-cloud/orca_organizer-feature-02-fill-en.png` | 1600x900 |
| Gallery 3, "Set spacing and rotation" | `assets/orca-cloud/orca_organizer-feature-03-spacing-en.png` | 1600x900 |
| Gallery 4, "Review the arranged plate" | `assets/orca-cloud/orca_organizer-feature-04-review-en.png` | 1600x900 |

The description embeds the four gallery images from `raw.githubusercontent.com` on `main`, so they show there
once the commit that adds them is merged.

The GitHub social preview (**Settings > General > Social preview**) is
`assets/social/orca_organizer-social-preview-en.png` (1280x640).

The `-pt-BR` gallery and `assets/readme/` images are kept only for README.pt-BR.md.

To rebuild any of these, see "Listing and README images" in [development.md](development.md).
`assets/marketing/asset-manifest.json` records how each file was made.
