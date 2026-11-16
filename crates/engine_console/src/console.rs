//! Логика консоли: реестр команд, разбор строки, выполнение, дополнение, журнал и история.
//! Интерфейса здесь нет — его рисует `ui.rs`, а строки можно выполнять и из кода
//! ([`Console::submit`]).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use engine_core::{AnyCVar, CHEATS, CVarFlags, CVarSource, CVars};
use engine_ecs::prelude::*;

/// Результат команды: текст для журнала (может быть пустым) или текст ошибки.
pub type CommandResult = Result<String, String>;

// [Rust] `Arc<dyn Fn ...>` — общий указатель на «что-то вызываемое» с такой сигнатурой
// (аналог `std::shared_ptr<std::function<...>>`). `Send + Sync` — можно хранить в ресурсе ECS.
type Handler = Arc<dyn Fn(&mut World, &[String]) -> CommandResult + Send + Sync>;

/// Команда консоли.
///
/// ```ignore
/// app.add_console_command(
///     ConsoleCommand::new("spawn", "Создать объект", |world, args| { /* ... */ Ok(String::new()) })
///         .with_usage("spawn <модель>")
///         .cheat(),
/// );
/// ```
#[derive(Clone)]
pub struct ConsoleCommand {
    name: &'static str,
    description: &'static str,
    usage: &'static str,
    cheat: bool,
    handler: Handler,
}

impl ConsoleCommand {
    /// `handler` получает мир ECS и аргументы (без имени команды).
    pub fn new(
        name: &'static str,
        description: &'static str,
        handler: impl Fn(&mut World, &[String]) -> CommandResult + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            description,
            usage: name,
            cheat: false,
            handler: Arc::new(handler),
        }
    }

    /// Как вызывать, для справки: `"reset <переменная>"`.
    #[must_use]
    pub fn with_usage(mut self, usage: &'static str) -> Self {
        self.usage = usage;
        self
    }

    /// Выполняется только при `cheats = true`.
    #[must_use]
    pub fn cheat(mut self) -> Self {
        self.cheat = true;
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn description(&self) -> &'static str {
        self.description
    }

    pub fn usage(&self) -> &'static str {
        self.usage
    }

    pub fn is_cheat(&self) -> bool {
        self.cheat
    }
}

impl std::fmt::Debug for ConsoleCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsoleCommand")
            .field("name", &self.name)
            .field("cheat", &self.cheat)
            .finish_non_exhaustive()
    }
}

/// Вид строки журнала — от него зависит цвет.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    /// Введённая строка.
    Input,
    Output,
    Error,
}

#[derive(Debug, Clone)]
pub struct ConsoleLine {
    pub kind: LineKind,
    pub text: String,
}

/// Результат дополнения по Tab.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Completion {
    /// Новая строка ввода; `None` — дополнять нечего (нет вариантов или их несколько
    /// и общая часть уже введена).
    pub text: Option<String>,
    /// Подходящие имена команд и переменных.
    pub candidates: Vec<&'static str>,
}

const MAX_LINES: usize = 1000;
const MAX_HISTORY: usize = 100;

/// Консоль (ресурс ECS): команды, журнал, история, очередь строк на выполнение.
#[derive(Resource, Default, Debug)]
pub struct Console {
    commands: BTreeMap<&'static str, ConsoleCommand>,
    lines: VecDeque<ConsoleLine>,
    history: Vec<String>,
    pending: Vec<String>,
}

impl Console {
    /// Регистрирует команду. Имя команды важнее имени переменной.
    pub fn add_command(&mut self, command: ConsoleCommand) {
        let previous = self.commands.insert(command.name, command);
        assert!(
            previous.is_none(),
            "две команды консоли с одним именем «{}»",
            previous.map_or("", |c| c.name)
        );
    }

    pub fn command(&self, name: &str) -> Option<&ConsoleCommand> {
        self.commands.get(name)
    }

    /// Все команды в алфавитном порядке.
    pub fn commands(&self) -> impl Iterator<Item = &ConsoleCommand> {
        self.commands.values()
    }

    /// Поставить строку в очередь: выполнится в стадии `Update` (как введённая вручную).
    pub fn submit(&mut self, line: impl Into<String>) {
        self.pending.push(line.into());
    }

    pub(crate) fn take_pending(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending)
    }

    /// Добавить текст в журнал; многострочный — по строке на запись.
    pub fn print(&mut self, kind: LineKind, text: impl AsRef<str>) {
        for line in text.as_ref().lines() {
            if self.lines.len() == MAX_LINES {
                self.lines.pop_front();
            }
            self.lines.push_back(ConsoleLine {
                kind,
                text: line.to_owned(),
            });
        }
    }

    pub fn lines(&self) -> &VecDeque<ConsoleLine> {
        &self.lines
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Введённые строки, от старых к новым.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    fn remember(&mut self, line: &str) {
        if self.history.last().is_some_and(|last| last == line) {
            return; // повтор подряд не засоряет историю
        }
        if self.history.len() == MAX_HISTORY {
            self.history.remove(0);
        }
        self.history.push(line.to_owned());
    }

    /// Дополнение последнего слова строки именем команды или переменной.
    pub fn complete(&self, cvars: &CVars, input: &str) -> Completion {
        // Начало последнего слова (байтовый индекс — после пробела)
        let start = input
            .char_indices()
            .rev()
            .find(|(_, c)| c.is_whitespace())
            .map_or(0, |(i, c)| i + c.len_utf8());
        let (head, word) = input.split_at(start);
        let mut candidates: Vec<&'static str> = self
            .commands
            .keys()
            .copied()
            .chain(cvars.iter().map(AnyCVar::name))
            .filter(|name| name.starts_with(word))
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        let text = match candidates.as_slice() {
            [] => None,
            [only] => Some(format!("{head}{only} ")),
            [first, rest @ ..] => {
                let common = rest
                    .iter()
                    .fold(*first, |prefix, name| common_prefix(prefix, name));
                (common.len() > word.len()).then(|| format!("{head}{common}"))
            }
        };
        Completion { text, candidates }
    }
}

fn common_prefix<'a>(a: &'a str, b: &str) -> &'a str {
    let len = a
        .char_indices()
        .zip(b.chars())
        .find(|((_, x), y)| x != y)
        .map_or(a.len().min(b.len()), |((i, _), _)| i);
    &a[..len]
}

/// Делит строку на слова по пробелам. Кавычки объединяют слова: `sign "Привет, мир"`;
/// `\"` внутри кавычек — сама кавычка.
pub fn tokenize(line: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    // Есть ли начатое слово: `""` — пустой аргумент, а не его отсутствие
    let mut started = false;
    let mut quoted = false;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            '\\' if quoted => {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            c if c.is_whitespace() && !quoted => {
                if started {
                    tokens.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            c => {
                current.push(c);
                started = true;
            }
        }
    }
    if quoted {
        return Err("не закрыта кавычка".to_owned());
    }
    if started {
        tokens.push(current);
    }
    Ok(tokens)
}

/// Делит строку на команды по `;` вне кавычек: `cheats true; sun 0.5`.
pub fn split_commands(line: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut quoted = false;
    let mut escaped = false;
    let mut start = 0;
    for (i, c) in line.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            ';' if !quoted => {
                parts.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&line[start..]);
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect()
}

/// Выполняет строку: команды через `;`, каждая — команда или чтение/изменение
/// переменной. Результат — в журнал; в историю строка попадает целиком.
pub fn execute(world: &mut World, line: &str) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    world.resource_mut::<Console>().remember(line);
    for part in split_commands(line) {
        execute_one(world, part);
    }
}

fn execute_one(world: &mut World, line: &str) {
    tracing::info!("Консоль: {line}");
    let mut console = world.resource_mut::<Console>();
    console.print(LineKind::Input, format!("> {line}"));
    let tokens = match tokenize(line) {
        Ok(tokens) => tokens,
        Err(error) => {
            console.print(LineKind::Error, error);
            return;
        }
    };
    let Some((name, args)) = tokens.split_first() else {
        return;
    };
    // Команда копируется (это счётчик ссылок `Arc`): обработчику нужен весь мир,
    // включая сам ресурс консоли, — заимствование ресурса на время вызова держать нельзя
    let command = console.command(name).cloned();
    let result = match command {
        Some(command) if command.cheat && !CHEATS.get() => {
            Err(format!("«{name}» — чит-команда, сначала включите cheats"))
        }
        Some(command) => (command.handler)(world, args),
        None => cvar_command(world, name, args),
    };
    let mut console = world.resource_mut::<Console>();
    match result {
        Ok(text) => console.print(LineKind::Output, text),
        Err(error) => console.print(LineKind::Error, error),
    }
}

/// `имя` — показать переменную, `имя значение` — изменить.
fn cvar_command(world: &mut World, name: &str, args: &[String]) -> CommandResult {
    let mut cvars = world.resource_mut::<CVars>();
    let Some(cvar) = cvars.get(name) else {
        return Err(unknown_name(world, name));
    };
    if args.is_empty() {
        return Ok(describe(cvar));
    }
    // Строковое значение можно писать и без кавычек: слова склеиваются через пробел
    cvars
        .set(name, &args.join(" "), CVarSource::Console)
        .map_err(|error| error.to_string())?;
    Ok(format!("{name} = {}", cvar.value_text()))
}

/// Ошибка «нет такого имени» с подсказкой похожих.
pub(crate) fn unknown_name(world: &World, name: &str) -> String {
    let console = world.resource::<Console>();
    let cvars = world.resource::<CVars>();
    let similar: Vec<&str> = console
        .commands
        .keys()
        .copied()
        .chain(cvars.iter().map(AnyCVar::name))
        .filter(|candidate| candidate.contains(name) || name.contains(*candidate))
        .take(5)
        .collect();
    if similar.is_empty() {
        format!("неизвестная команда или переменная «{name}» (help — список)")
    } else {
        format!(
            "неизвестная команда или переменная «{name}». Возможно: {}",
            similar.join(", ")
        )
    }
}

/// Описание переменной: значение, значение по умолчанию, тип, флаги, описание.
pub(crate) fn describe(cvar: &dyn AnyCVar) -> String {
    let mut flags = String::new();
    if cvar.flags().contains(CVarFlags::STARTUP) {
        flags.push_str(", только при запуске");
    }
    if cvar.flags().contains(CVarFlags::CHEAT) {
        flags.push_str(", чит");
    }
    format!(
        "{} = {} (по умолчанию {}, {}{flags})\n  {}",
        cvar.name(),
        cvar.value_text(),
        cvar.default_text(),
        cvar.type_name(),
        cvar.description()
    )
}

#[cfg(test)]
mod tests {
    use engine_core::CVar;

    use super::*;

    fn world() -> World {
        let mut world = World::new();
        world.init_resource::<CVars>();
        world.init_resource::<Console>();
        world
    }

    fn last(world: &World) -> ConsoleLine {
        world.resource::<Console>().lines().back().cloned().unwrap()
    }

    #[test]
    fn tokenize_quotes_and_escapes() {
        assert_eq!(tokenize("  a  b ").unwrap(), ["a", "b"]);
        assert_eq!(
            tokenize(r#"sign "Привет, мир" x"#).unwrap(),
            ["sign", "Привет, мир", "x"]
        );
        assert_eq!(tokenize(r#"say "a \"b\"""#).unwrap(), ["say", r#"a "b""#]);
        assert_eq!(
            tokenize(r#"set """#).unwrap(),
            ["set", ""],
            "пустой аргумент"
        );
        assert!(tokenize(r#"say "oops"#).is_err());
    }

    #[test]
    fn split_by_semicolon_outside_quotes() {
        assert_eq!(split_commands("a 1; b ;; c"), ["a 1", "b", "c"]);
        assert_eq!(
            split_commands(r#"sign "a; b"; sun 1"#),
            [r#"sign "a; b""#, "sun 1"]
        );
        assert_eq!(split_commands(r#"say "x\"; y""#), [r#"say "x\"; y""#]);
    }

    #[test]
    fn show_and_set_cvar() {
        static SPEED: CVar<f32> = CVar::new("test.console.speed", 1.0, "Скорость");
        let mut world = world();
        world.resource_mut::<CVars>().register(&SPEED);
        execute(&mut world, "test.console.speed");
        assert!(
            world
                .resource::<Console>()
                .lines()
                .iter()
                .any(|l| l.text.contains("по умолчанию 1"))
        );
        execute(&mut world, "test.console.speed 2.5");
        assert!((SPEED.get() - 2.5).abs() < f32::EPSILON);
        execute(&mut world, "test.console.speed много");
        assert_eq!(last(&world).kind, LineKind::Error);
        assert!(
            (SPEED.get() - 2.5).abs() < f32::EPSILON,
            "значение не испорчено"
        );
        assert_eq!(world.resource::<Console>().history().len(), 3);
    }

    #[test]
    fn commands_get_args_and_world() {
        #[derive(Resource, Default)]
        struct Said(Vec<String>);
        let mut world = world();
        world.init_resource::<Said>();
        world
            .resource_mut::<Console>()
            .add_command(ConsoleCommand::new("say", "", |world, args| {
                world.resource_mut::<Said>().0.extend_from_slice(args);
                Ok(format!("сказано {}", args.len()))
            }));
        execute(&mut world, r#"say "один два" три"#);
        assert_eq!(world.resource::<Said>().0, ["один два", "три"]);
        assert_eq!(last(&world).text, "сказано 2");
    }

    #[test]
    fn cheat_command_needs_cheats() {
        // CHEATS — общая глобальная переменная; другие тесты её не трогают
        let mut world = world();
        world
            .resource_mut::<Console>()
            .add_command(ConsoleCommand::new("god", "", |_, _| Ok("ок".to_owned())).cheat());
        execute(&mut world, "god");
        assert_eq!(last(&world).kind, LineKind::Error);
    }

    #[test]
    fn unknown_name_suggests_similar() {
        static FOG: CVar<bool> = CVar::new("test.console.fog", false, "");
        let mut world = world();
        world.resource_mut::<CVars>().register(&FOG);
        execute(&mut world, "fog");
        let line = last(&world);
        assert_eq!(line.kind, LineKind::Error);
        assert!(line.text.contains("test.console.fog"), "{}", line.text);
    }

    #[test]
    fn completion_extends_common_prefix() {
        static A: CVar<bool> = CVar::new("test.complete.alpha", false, "");
        static B: CVar<bool> = CVar::new("test.complete.beta", false, "");
        let mut cvars = CVars::default();
        cvars.register(&A);
        cvars.register(&B);
        let mut console = Console::default();
        console.add_command(ConsoleCommand::new("toggle", "", |_, _| Ok(String::new())));

        // Общая часть двух имён
        let c = console.complete(&cvars, "test.co");
        assert_eq!(c.text.as_deref(), Some("test.complete."));
        assert_eq!(c.candidates.len(), 2);
        // Общая часть уже введена — дополнять нечего, только варианты
        assert_eq!(console.complete(&cvars, "test.complete.").text, None);
        // Единственный вариант — целиком и пробел; дополняется последнее слово
        assert_eq!(
            console
                .complete(&cvars, "toggle test.complete.b")
                .text
                .as_deref(),
            Some("toggle test.complete.beta ")
        );
        assert_eq!(
            console.complete(&cvars, "tog").text.as_deref(),
            Some("toggle ")
        );
        assert_eq!(console.complete(&cvars, "zzz"), Completion::default());
    }

    #[test]
    fn history_skips_repeats_and_is_bounded() {
        let mut console = Console::default();
        console.remember("a");
        console.remember("a");
        console.remember("b");
        assert_eq!(console.history(), ["a", "b"]);
        for i in 0..200 {
            console.remember(&i.to_string());
        }
        assert_eq!(console.history().len(), MAX_HISTORY);
        assert_eq!(console.history().last().unwrap(), "199");
    }
}
