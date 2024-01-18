plugins {
    kotlin("jvm")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))

//    implementation("org.jetbrains.kotlinx:multik-core:0.2.2")
//    implementation("org.jetbrains.kotlinx:multik-default:0.2.2")
//    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-tensorflow:0.5.2")

    implementation(project(":kotlin-dl-utils"))
    implementation(project(":kotlin-gym"))

    testImplementation("org.jetbrains.kotlin:kotlin-test")
}

tasks.test {
    useJUnitPlatform()
}
