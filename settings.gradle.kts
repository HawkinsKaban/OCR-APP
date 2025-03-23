pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // JitPack for PDF viewer
        maven {
            url = uri("https://jitpack.io")
            content {
                includeGroup("com.github.afreakyelf")
            }
        }
        // Maven Central and others for OpenCV
        maven { url = uri("https://repo1.maven.org/maven2/") }
    }
}

rootProject.name = "OCR TFLite"
include(":app")