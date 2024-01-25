plugins {
    kotlin("jvm")
}



repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))

    api("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
}
