plugins {
    kotlin("jvm")
}

repositories {
    mavenCentral()
}

dependencies {
    api("org.deeplearning4j:deeplearning4j-core:1.0.0-M2.1")
    api("org.nd4j:nd4j-native-platform:1.0.0-M2.1")

    testImplementation("org.jetbrains.kotlin:kotlin-test")
}

tasks.test {
    useJUnitPlatform()
}
