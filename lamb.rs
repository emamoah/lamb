// ,---@>
//  W-W'
// rustc -o lamb lamb.rs
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::{
    borrow::Cow,
    env,
    fs::{self, File},
    io::{self, BufRead, ErrorKind, Read, Write},
    process,
    rc::Rc,
    sync::Mutex,
};

const VEC_INIT_CAP: usize = 256;
fn vec_reserve<T>(vec: &mut Vec<T>) {
    if vec.capacity() == 0 {
        vec.reserve(VEC_INIT_CAP);
    }
}
fn vec_append<T>(vec: &mut Vec<T>, val: T) {
    vec_reserve(vec);
    vec.push(val);
}

fn stringify(bytes: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(bytes)
}

type Cmd = Vec<String>;

#[cfg(not(target_os = "windows"))]
fn cmd_run(cmd: &Cmd) -> bool {
    if cmd.is_empty() {
        eprintln!("ERROR: Could not run empty command");
        return false;
    }

    let mut child = match process::Command::new(&cmd[0]).args(&cmd[1..]).spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!(
                "ERROR: Could not spawn child process for {}: {}",
                &cmd[0], e
            );
            return false;
        }
    };

    match child.wait() {
        Err(e) => {
            eprintln!(
                "ERROR: Could not wait on command (pid: {}): {}",
                child.id(),
                e
            );
            return false;
        }
        Ok(status) => {
            if !status.success() {
                match status.code() {
                    Some(code) => {
                        eprintln!("ERROR: Command exited with code {code}");
                    }
                    None => {
                        #[cfg(unix)]
                        eprint!(
                            "ERROR: Command process was terminated by a signal {}",
                            status.signal().unwrap()
                        );
                    }
                }

                return false;
            }
        }
    }

    true
}

// RETURNS:
//  0 - file does not exist
//  1 - file exists
// -1 - error while checking if file exists. The error is logged
fn file_exists(file_path: &String) -> i32 {
    match fs::metadata(file_path) {
        Err(e) => {
            if e.kind() == ErrorKind::NotFound {
                return 0;
            }

            eprintln!("ERROR: Could not check if file {file_path} exists: {e}");
            -1
        }
        Ok(_) => 1,
    }
}

fn read_entire_file(path: &String, sb: &mut Vec<u8>) -> bool {
    if let Err(e) = File::options()
        .read(true)
        .open(path)
        .and_then(|mut f| f.read_to_end(sb))
    {
        eprintln!("ERROR: Could not read file {path}: {e}");
        return false;
    }

    true
}

fn write_entire_file(path: &String, data: &mut [u8]) -> bool {
    let mut f = match File::options()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: Could not open file {path} for writing: {e}");
            return false;
        }
    };

    if let Err(e) = f.write_all(data) {
        eprintln!("ERROR: Could not write into file {path}: {e}");
        return false;
    }

    true
}

static LABELS: Mutex<Vec<&'static [u8]>> = Mutex::new(Vec::new());

fn intern_label(label: &[u8]) -> &'static [u8] {
    let mut labels = LABELS.lock().unwrap();
    for l in &*labels {
        if label == *l {
            return l;
        }
    }

    let result = label.to_owned().leak();
    vec_append(&mut labels, result);

    result
}

#[derive(Debug, Copy, Clone)]
struct Symbol {
    /// Displayed name of the symbol.
    label: &'static [u8],
    /// Internal tag that makes two symbols with the same label different if needed.
    /// Usually used to obtain a fresh symbol for capture avoiding substitution.
    tag: usize,
}

impl Symbol {
    fn label(&self) -> Cow<'_, str> {
        stringify(self.label)
    }
}

fn symbol_eq(a: &Symbol, b: &Symbol) -> bool {
    // NOTE: We compare addresses of the labels because they are expected to be interned with intern_label()
    a.label.as_ptr() == b.label.as_ptr() && a.tag == b.tag
}

fn symbol(label: &[u8]) -> Symbol {
    Symbol {
        label: intern_label(label),
        tag: 0,
    }
}

fn symbol_fresh(mut s: Symbol) -> Symbol {
    static GLOBAL_COUNTER: Mutex<usize> = Mutex::new(0);
    let mut global_counter = GLOBAL_COUNTER.lock().unwrap();

    *global_counter += 1;
    s.tag = *global_counter;

    s
}

#[derive(Debug)]
struct ExprFun {
    param: Symbol,
    body: Rc<Expr>,
}

#[derive(Debug)]
struct ExprApp {
    lhs: Rc<Expr>,
    rhs: Rc<Expr>,
}

#[derive(Debug)]
enum Expr {
    Var(Symbol),
    Fun(ExprFun),
    App(ExprApp),
    Mag(&'static [u8]),
}

fn var(name: Symbol) -> Rc<Expr> {
    Rc::new(Expr::Var(name))
}

fn magic(label: &[u8]) -> Rc<Expr> {
    Rc::new(Expr::Mag(intern_label(label)))
}

fn fun(param: Symbol, body: Rc<Expr>) -> Rc<Expr> {
    Rc::new(Expr::Fun(ExprFun { param, body }))
}

fn app(lhs: Rc<Expr>, rhs: Rc<Expr>) -> Rc<Expr> {
    Rc::new(Expr::App(ExprApp { lhs, rhs }))
}

impl Expr {
    fn display(&self, sb: &mut Vec<u8>) {
        let mut expr = self;
        match expr {
            Expr::Var(var) => {
                write!(sb, "{}", var.label()).unwrap_or(());
                if var.tag > 0 {
                    write!(sb, ":{}", var.tag).unwrap_or(());
                }
            }
            Expr::Fun(_) => {
                write!(sb, "\\").unwrap_or(());
                while let Expr::Fun(fun) = expr {
                    if fun.param.tag > 0 {
                        write!(sb, "{}:{}.", fun.param.label(), fun.param.tag).unwrap_or(());
                    } else {
                        write!(sb, "{}.", fun.param.label()).unwrap_or(());
                    }
                    expr = &fun.body;
                }
                expr.display(sb);
            }
            Expr::App(app) => {
                let lhs = &app.lhs;
                let lhs_paren = matches!(**lhs, Expr::Fun(_));
                if lhs_paren {
                    sb.push(b'(');
                }
                lhs.display(sb);
                if lhs_paren {
                    sb.push(b')');
                }

                sb.push(b' ');

                let rhs = &app.rhs;
                let rhs_paren = !matches!(**rhs, Expr::Var(_) | Expr::Mag(_));
                if rhs_paren {
                    sb.push(b'(');
                }
                rhs.display(sb);
                if rhs_paren {
                    sb.push(b')');
                }
            }
            Expr::Mag(mag) => {
                write!(sb, "#{}", stringify(mag)).unwrap_or(());
            }
        }
    }

    fn dump_ast(&self) {
        static STACK: Mutex<Vec<bool>> = Mutex::new(Vec::new());

        fn push_stack(val: bool) {
            let mut stack = STACK.lock().unwrap();
            vec_append(&mut stack, val);
        }

        fn pop_stack() {
            STACK.lock().unwrap().pop();
        }

        {
            let stack = STACK.lock().unwrap();
            for i in 0..stack.len() {
                if i + 1 == stack.len() {
                    print!("+--");
                } else if stack[i] {
                    print!("|  ");
                } else {
                    print!("   ");
                }
            }
        }

        match self {
            Expr::Var(var) => {
                if var.tag == 0 {
                    println!("[VAR] {}", var.label());
                } else {
                    println!("[VAR] {}:{}", var.label(), var.tag);
                }
            }
            Expr::Fun(fun) => {
                if fun.param.tag == 0 {
                    println!("[FUN] \\{}", fun.param.label());
                } else {
                    println!("[FUN] \\{}:{}", fun.param.label(), fun.param.tag);
                }
                push_stack(false);
                fun.body.dump_ast();
                pop_stack();
            }
            Expr::App(app) => {
                println!("[APP]");

                push_stack(true);
                app.lhs.dump_ast();
                pop_stack();

                push_stack(false);
                app.rhs.dump_ast();
                pop_stack();
            }
            Expr::Mag(mag) => {
                println!("[MAG] #{}", stringify(mag));
            }
        }
    }

    fn trace(&self) {
        static SB: Mutex<Vec<u8>> = Mutex::new(Vec::new());
        let mut sb = SB.lock().unwrap();
        sb.clear();
        vec_reserve(&mut sb);
        self.display(&mut sb);
        io::stdout().write_all(&sb).unwrap_or(());
    }

    fn eval1(self: Rc<Expr>) -> Option<Rc<Expr>> {
        match self.as_ref() {
            Expr::Var(_) => Some(self.clone()),
            Expr::Fun(f) => {
                let body = f.body.clone().eval1()?;
                if !Rc::ptr_eq(&body, &f.body) {
                    Some(fun(f.param, body))
                } else {
                    Some(self)
                }
            }
            Expr::App(a) => {
                let lhs = a.lhs.clone();
                let rhs = a.rhs.clone();

                if let Expr::Fun(f) = lhs.as_ref() {
                    return Some(replace(f.param, f.body.clone(), rhs));
                } else if let Expr::Mag(m) = lhs.as_ref() {
                    if *m == intern_label(b"trace") {
                        let new_rhs = rhs.clone().eval1()?;
                        if Rc::ptr_eq(&new_rhs, &rhs) {
                            print!("TRACE: ");
                            rhs.trace();
                            println!();
                            return Some(rhs);
                        }
                        return Some(app(lhs, new_rhs));
                    } else if *m == intern_label(b"void") {
                        let new_rhs = rhs.clone().eval1()?;
                        if Rc::ptr_eq(&new_rhs, &rhs) {
                            return Some(lhs);
                        }
                        return Some(app(lhs, new_rhs));
                    }
                    println!("ERROR: unknown magic #{}", stringify(m));
                    return None;
                }

                let new_lhs = lhs.clone().eval1()?;
                if !Rc::ptr_eq(&lhs, &new_lhs) {
                    return Some(app(new_lhs, rhs));
                }

                let new_rhs = rhs.clone().eval1()?;
                if !Rc::ptr_eq(&rhs, &new_rhs) {
                    return Some(app(lhs, new_rhs));
                }

                Some(self)
            }
            Expr::Mag(_) => Some(self),
        }
    }
}

fn is_var_free_there(name: &Symbol, there: &Rc<Expr>) -> bool {
    match there.as_ref() {
        Expr::Var(var) => symbol_eq(var, name),
        Expr::Fun(fun) => {
            if symbol_eq(&fun.param, name) {
                false
            } else {
                is_var_free_there(name, &fun.body)
            }
        }
        Expr::App(app) => {
            if is_var_free_there(name, &app.lhs) {
                return true;
            }
            if is_var_free_there(name, &app.rhs) {
                return true;
            }
            false
        }
        Expr::Mag(_) => false,
    }
}

fn replace(param: Symbol, body: Rc<Expr>, arg: Rc<Expr>) -> Rc<Expr> {
    match body.as_ref() {
        Expr::Mag(_) => body,
        Expr::Var(v) => {
            if symbol_eq(v, &param) {
                arg
            } else {
                body
            }
        }
        Expr::Fun(f) => {
            if symbol_eq(&f.param, &param) {
                return body;
            }
            if !is_var_free_there(&f.param, &arg) {
                return fun(f.param, replace(param, f.body.clone(), arg));
            }
            let fresh_param_name = symbol_fresh(f.param);
            let fresh_param = var(fresh_param_name);
            fun(
                fresh_param_name,
                replace(param, replace(f.param, f.body.clone(), fresh_param), arg),
            )
        }
        Expr::App(a) => app(
            replace(param, a.lhs.clone(), arg.clone()),
            replace(param, a.rhs.clone(), arg.clone()),
        ),
    }
}

#[derive(Copy, Clone, PartialEq)]
enum Token {
    Invalid,
    End,
    Oparen,
    Cparen,
    Lambda,
    Dot,
    Colon,
    Semicolon,
    Equals,
    Name,
    Magic,
}

impl Token {
    // We could derive `Debug`, but just to stay consistent with the original...
    fn display(&self) -> &str {
        match self {
            Token::Invalid => "TOKEN_INVALID",
            Token::End => "TOKEN_END",
            Token::Oparen => "TOKEN_OPAREN",
            Token::Cparen => "TOKEN_CPAREN",
            Token::Lambda => "TOKEN_LAMBDA",
            Token::Dot => "TOKEN_DOT",
            Token::Colon => "TOKEN_COLON",
            Token::Semicolon => "TOKEN_SEMICOLON",
            Token::Equals => "TOKEN_EQUALS",
            Token::Name => "TOKEN_NAME",
            Token::Magic => "TOKEN_MAGIC",
        }
    }
}

#[derive(Copy, Clone)]
struct Cur {
    pos: usize,
    bol: usize,
    row: usize,
}

struct Lexer<'a> {
    content: &'a mut [u8],
    file_path: Option<&'a String>,

    cur: Cur,

    token: Token,
    string: Vec<u8>,
    row: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    fn count(&self) -> usize {
        self.content.len()
    }

    fn init(content: &'a mut [u8], file_path: Option<&'a String>) -> Self {
        Self {
            content,
            file_path,

            cur: Cur {
                pos: 0,
                bol: 0,
                row: 0,
            },

            // Original C version initialised with 0. Same as first variant.
            token: Token::Invalid,
            string: Vec::with_capacity(VEC_INIT_CAP),
            row: 0,
            col: 0,
        }
    }

    fn print_loc(&self, stream: &mut impl io::Write) {
        if let Some(file_path) = &self.file_path {
            write!(stream, "{file_path}:").unwrap_or(());
        }
        write!(stream, "{}:{}: ", self.row, self.col).unwrap_or(());
    }

    fn curr_char(&self) -> u8 {
        if self.cur.pos >= self.count() {
            return 0;
        }

        self.content[self.cur.pos]
    }

    fn next_char(&mut self) -> u8 {
        if self.cur.pos >= self.count() {
            return 0;
        }
        let x = self.content[self.cur.pos];
        self.cur.pos += 1;
        if x == b'\n' {
            self.cur.row += 1;
            self.cur.bol = self.cur.pos;
        }

        x
    }

    fn trim_left(&mut self) {
        while self.curr_char().is_ascii_whitespace() {
            self.next_char();
        }
    }

    fn starts_with(&self, prefix: &[u8]) -> bool {
        self.content[self.cur.pos..].starts_with(prefix)
    }

    fn drop_line(&mut self) {
        while self.cur.pos < self.count() && self.next_char() != b'\n' {}
    }

    fn next(&mut self) -> bool {
        loop {
            self.trim_left();
            if self.starts_with(b"//") {
                self.drop_line();
            } else {
                break;
            }
        }

        self.row = self.cur.row + 1;
        self.col = self.cur.pos - self.cur.bol + 1;

        let mut x = self.next_char();
        if x == b'\0' {
            self.token = Token::End;
            return true;
        }

        match x {
            b'(' => {
                self.token = Token::Oparen;
                return true;
            }
            b')' => {
                self.token = Token::Cparen;
                return true;
            }
            b'\\' => {
                self.token = Token::Lambda;
                return true;
            }
            b'.' => {
                self.token = Token::Dot;
                return true;
            }
            b':' => {
                self.token = Token::Colon;
                return true;
            }
            b';' => {
                self.token = Token::Semicolon;
                return true;
            }
            b'=' => {
                self.token = Token::Equals;
                return true;
            }
            _ => {}
        }

        if x == b'#' {
            self.token = Token::Magic;
            self.string.clear();
            while issymbol(self.curr_char()) {
                x = self.next_char();
                self.string.push(x);
            }
            return true;
        }

        if issymbol(x) {
            self.token = Token::Name;
            self.string.clear();
            self.string.push(x);
            while issymbol(self.curr_char()) {
                x = self.next_char();
                self.string.push(x);
            }
            return true;
        }

        self.token = Token::Invalid;
        self.print_loc(&mut io::stderr());
        eprintln!("ERROR: Unknown token starts with `{x}`");

        false
    }

    fn peek(&mut self) -> bool {
        let cur = self.cur;
        let result = self.next();
        self.cur = cur;

        result
    }

    fn report_unexpected(&self, expected: Token) {
        self.print_loc(&mut io::stderr());
        eprintln!(
            "ERROR: Unexpected token {}. Expected {} instead.",
            self.token.display(),
            expected.display()
        );
    }

    fn expect(&mut self, expected: Token) -> bool {
        if !self.next() {
            return false;
        }
        if self.token != expected {
            self.report_unexpected(expected);
            return false;
        }

        true
    }

    fn parse_fun(&mut self) -> Option<Rc<Expr>> {
        if !self.expect(Token::Name) {
            return None;
        }
        let arg = symbol(&self.string);
        if !self.expect(Token::Dot) {
            return None;
        }

        let a: Token;
        let b: Token;
        let cur = self.cur;

        if !self.next() {
            return None;
        }
        a = self.token;
        if !self.next() {
            return None;
        }
        b = self.token;

        self.cur = cur;

        let body = if a == Token::Name && b == Token::Dot {
            self.parse_fun()?
        } else {
            self.parse_expr()?
        };

        Some(fun(arg, body))
    }

    fn parse_primary(&mut self) -> Option<Rc<Expr>> {
        if !self.next() {
            return None;
        }
        match self.token {
            Token::Oparen => {
                let expr = self.parse_expr()?;
                if !self.expect(Token::Cparen) {
                    return None;
                }
                Some(expr)
            }
            Token::Lambda => self.parse_fun(),
            Token::Magic => Some(magic(&self.string)),
            Token::Name => Some(var(symbol(&self.string))),
            token => {
                self.print_loc(&mut io::stderr());
                eprintln!(
                    "ERROR: Unexpected token {}. Expected a primary expression instead.",
                    token.display()
                );
                None
            }
        }
    }

    fn parse_expr(&mut self) -> Option<Rc<Expr>> {
        let mut expr = self.parse_primary()?;

        if !self.peek() {
            return None;
        }
        while self.token != Token::Cparen
            && self.token != Token::End
            && self.token != Token::Semicolon
        {
            let rhs = self.parse_primary()?;
            expr = app(expr, rhs);
            if !self.peek() {
                return None;
            }
        }

        Some(expr)
    }
}

fn issymbol(x: u8) -> bool {
    x.is_ascii_alphanumeric() || x == b'_'
}

struct Command {
    name: &'static [u8],
    signature: &'static [u8],
    description: &'static [u8],
}

type Commands = Vec<Command>;

fn command(
    commands: &mut Commands,
    input: &[u8],
    name: &'static [u8],
    signature: &'static [u8],
    description: &'static [u8],
) -> bool {
    vec_append(
        commands,
        Command {
            name,
            signature,
            description,
        },
    );

    name.starts_with(input)
}

fn print_available_commands(commands: &Commands) {
    println!("Available commands:");
    let mut max_name_width = 0;
    let mut max_sig_width = 0;

    for command in commands {
        let name_width = command.name.len();
        let sig_width = command.signature.len();
        if name_width > max_name_width {
            max_name_width = name_width;
        }
        if sig_width > max_sig_width {
            max_sig_width = sig_width;
        }
    }

    for command in commands {
        println!(
            "  :{:<max_name_width$} {:<max_sig_width$} - {}",
            stringify(command.name),
            stringify(command.signature),
            stringify(command.description)
        );
    }
}

struct Binding {
    name: Symbol,
    body: Rc<Expr>,
}

type Bindings = Vec<Binding>;

fn create_binding(bindings: &mut Bindings, name: Symbol, body: Rc<Expr>) {
    for binding in &mut *bindings {
        if symbol_eq(&binding.name, &name) {
            binding.body = body;
            if name.tag == 0 {
                println!("Updated binding {}", name.label());
            } else {
                println!("Updated binding {}:{}", name.label(), name.tag);
            }
            return;
        }
    }
    let binding = Binding { name, body };
    vec_append(bindings, binding);
    if name.tag == 0 {
        println!("Created binding {}", name.label());
    } else {
        println!("Created binding {}:{}", name.label(), name.tag);
    }
}

fn create_bindings_from_file(file_path: &String, bindings: &mut Bindings) -> bool {
    static SB: Mutex<Vec<u8>> = Mutex::new(Vec::new());
    let mut sb = SB.lock().unwrap();
    let mut l: Lexer;

    sb.clear();
    if !read_entire_file(file_path, &mut sb) {
        return false;
    }

    l = Lexer::init(&mut sb, Some(file_path));

    if !l.peek() {
        return false;
    }
    while l.token != Token::End {
        if !l.expect(Token::Name) {
            return false;
        }
        let name = symbol(&l.string);
        if !l.expect(Token::Equals) {
            return false;
        }
        let Some(body) = l.parse_expr() else {
            return false;
        };
        if !l.expect(Token::Semicolon) {
            return false;
        }
        create_binding(bindings, name, body);
        if !l.peek() {
            return false;
        }
    }

    true
}

fn replace_active_file_path_from_lexer_if_not_empty(l: &Lexer, active_file_path: &mut String) {
    let path_data = l.content[l.cur.pos..].trim_ascii();

    if !path_data.is_empty() {
        *active_file_path = stringify(path_data).into();
    }
}

fn main() {
    let mut buffer: Vec<u8> = Vec::with_capacity(1024);
    let mut commands: Commands = Vec::new();
    let mut bindings: Bindings = Vec::new();
    let mut l: Lexer;

    // TODO: implement SIGINT handling

    #[cfg(not(target_os = "windows"))]
    let editor = env::var("LAMB_EDITOR")
        .or_else(|_| env::var("EDITOR"))
        .unwrap_or("vi".into());

    let mut active_file_path = String::new();

    let argc = env::args().len();
    if argc == 2 {
        active_file_path = env::args().nth(1).unwrap();
    } else if argc > 2 {
        eprintln!("ERROR: only a single active file is supported right now");
        process::exit(1);
    }

    if !active_file_path.is_empty() {
        create_bindings_from_file(&active_file_path, &mut bindings);
    }

    println!(",---@>");
    println!(" W-W'");
    println!("Enter :help for more info");

    'again: loop {
        print!("@> ");
        io::stdout().flush().unwrap_or(());
        buffer.clear();
        match io::stdin().lock().read_until(b'\n', &mut buffer) {
            Ok(count) => {
                if count == 0 {
                    break 'again;
                }
            }
            Err(_) => {
                println!();
                continue 'again;
            }
        }
        let source = &mut buffer;

        l = Lexer::init(source, None);

        if !l.peek() {
            continue 'again;
        }
        if l.token == Token::End {
            continue 'again;
        }
        if l.token == Token::Colon {
            if !l.next() {
                continue 'again;
            }
            if !l.expect(Token::Name) {
                continue 'again;
            }

            commands.clear();

            if command(
                &mut commands,
                &l.string,
                b"load",
                b"[path]",
                b"Load/reload bindings from a file.",
            ) {
                replace_active_file_path_from_lexer_if_not_empty(&l, &mut active_file_path);
                if active_file_path.is_empty() {
                    eprintln!("ERROR: No active file to reload from. Do `:load <path>`.");
                    continue 'again;
                }

                bindings.clear();
                create_bindings_from_file(&active_file_path, &mut bindings);
                continue 'again;
            }

            if command(
                &mut commands,
                &l.string,
                b"save",
                b"[path]",
                b"Save current bindings to a file.",
            ) {
                replace_active_file_path_from_lexer_if_not_empty(&l, &mut active_file_path);
                if active_file_path.is_empty() {
                    eprintln!("ERROR: No active file to save to. Do `:save <path>`.");
                    continue 'again;
                }

                static SB: Mutex<Vec<u8>> = Mutex::new(Vec::new());
                let mut sb = SB.lock().unwrap();
                sb.clear();
                for binding in &bindings {
                    assert!(binding.name.tag == 0);
                    vec_reserve(&mut sb);
                    write!(sb, "{} = ", binding.name.label()).unwrap_or(());
                    binding.body.display(sb.as_mut());
                    writeln!(sb, ";").unwrap_or(());
                }

                let exists = file_exists(&active_file_path);
                if exists < 0 {
                    continue 'again;
                }
                if exists > 0 {
                    print!(
                        "WARNING! This command will override the formatting of {active_file_path}. Really save? [N/y] "
                    );
                    io::stdout().flush().unwrap_or(());
                    buffer.clear();
                    match io::stdin().lock().read_until(b'\n', &mut buffer) {
                        Ok(count) => {
                            if count == 0 {
                                break 'again;
                            }
                        }
                        Err(_) => {
                            println!();
                            continue 'again;
                        }
                    }
                    if buffer[0] != b'y' && buffer[0] != b'Y' {
                        continue 'again;
                    }
                }

                if !write_entire_file(&active_file_path, &mut sb) {
                    continue 'again;
                }
                println!("Saved all the bindings to {active_file_path}");
                continue 'again;
            }

            #[cfg(target_os = "windows")]
            if command(
                &mut commands,
                &l.string,
                b"edit",
                b"[path]",
                b"Edit current active file. Reload it on exit.",
            ) {
                eprintln!("TODO: editing files is not implemented on Windows yet! Sorry!");
            }

            #[cfg(not(target_os = "windows"))]
            if command(
                &mut commands,
                &l.string,
                b"edit",
                b"[path]",
                b"Edit current active file. Reload it on exit.",
            ) {
                replace_active_file_path_from_lexer_if_not_empty(&l, &mut active_file_path);
                if active_file_path.is_empty() {
                    eprintln!("ERROR: No active file to edit. Do `:edit <path>`.");
                    continue 'again;
                }

                static CMD: Mutex<Cmd> = Mutex::new(Vec::new());
                let mut cmd = CMD.lock().unwrap();
                cmd.clear();
                vec_append(&mut cmd, editor.clone());
                vec_append(&mut cmd, active_file_path.clone());
                if cmd_run(&cmd) {
                    bindings.clear();
                    create_bindings_from_file(&active_file_path, &mut bindings);
                }

                continue 'again;
            }

            if command(
                &mut commands,
                &l.string,
                b"list",
                b"[names...]",
                b"list the bindings",
            ) {
                static SB: Mutex<Vec<u8>> = Mutex::new(Vec::new());
                static ARGS: Mutex<Vec<&'static [u8]>> = Mutex::new(Vec::new());
                let mut sb = SB.lock().unwrap();
                let mut args = ARGS.lock().unwrap();

                vec_reserve(&mut sb);

                args.clear();
                if !l.next() {
                    continue 'again;
                }
                while l.token == Token::Name {
                    vec_append(&mut args, intern_label(&l.string));
                    if !l.next() {
                        continue 'again;
                    }
                }
                if l.token != Token::End {
                    l.report_unexpected(Token::Name);
                    continue 'again;
                }

                if args.is_empty() {
                    for binding in &bindings {
                        assert!(binding.name.tag == 0);
                        sb.clear();
                        write!(sb, "{} = ", binding.name.label()).unwrap_or(());
                        binding.body.display(&mut sb);
                        writeln!(sb, ";").unwrap_or(());
                        io::stdout().write_all(&sb).unwrap_or(());
                    }
                    continue 'again;
                }

                for label in args.iter() {
                    let mut found = false;
                    for binding in &bindings {
                        assert!(binding.name.tag == 0);
                        if binding.name.label == *label {
                            sb.clear();
                            write!(sb, "{} = ", binding.name.label()).unwrap_or(());
                            binding.body.display(&mut sb);
                            writeln!(sb, ";").unwrap_or(());
                            io::stdout().write_all(&sb).unwrap_or(());
                            found = true;
                            break; // Maybe?
                        }
                    }
                    if !found {
                        eprintln!("ERROR: binding {} does not exist", stringify(label));
                        continue 'again;
                    }
                }

                continue 'again;
            }

            if command(
                &mut commands,
                &l.string,
                b"delete",
                b"<name>",
                b"delete a binding by name",
            ) {
                if !l.expect(Token::Name) {
                    continue 'again;
                }
                let name = symbol(&l.string);
                for (i, binding) in bindings.iter().enumerate() {
                    if symbol_eq(&binding.name, &name) {
                        bindings.remove(i);
                        println!("Deleted binding {}", name.label());
                        continue 'again;
                    }
                }
                println!("ERROR: binding {} was not found", name.label());
                continue 'again;
            }

            if command(
                &mut commands,
                &l.string,
                b"debug",
                b"<expr>",
                b"Step debug the evaluation of an expression",
            ) {
                let Some(mut expr) = l.parse_expr() else {
                    continue 'again;
                };
                if !l.expect(Token::End) {
                    continue 'again;
                }
                for binding in bindings.iter().rev() {
                    expr = replace(binding.name, expr, binding.body.clone());
                }

                loop {
                    // TODO: can't currently handle SIGINT

                    print!("DEBUG: ");
                    expr.trace();
                    println!();

                    print!("-> ");
                    io::stdout().flush().unwrap_or(());

                    // TODO: See original (https://github.com/tsoding/lamb/blob/main/lamb.c#L1306)
                    buffer.clear();
                    match io::stdin().lock().read_until(b'\n', &mut buffer) {
                        Ok(count) => {
                            if count == 0 {
                                break 'again;
                            }
                        }
                        Err(_) => {
                            println!();
                            continue 'again;
                        }
                    }

                    l = Lexer::init(&mut buffer, None);
                    if !l.next() {
                        continue 'again;
                    }
                    if l.token == Token::Name && l.string == b"quit" {
                        continue 'again;
                    }

                    let Some(expr1) = expr.clone().eval1() else {
                        continue 'again;
                    };
                    if Rc::ptr_eq(&expr, &expr1) {
                        break;
                    }
                    expr = expr1;
                }

                continue 'again;
            }

            if command(
                &mut commands,
                &l.string,
                b"ast",
                b"<expr>",
                b"print the AST of the expression",
            ) {
                let Some(expr) = l.parse_expr() else {
                    continue 'again;
                };
                if !l.expect(Token::End) {
                    continue 'again;
                }
                expr.dump_ast();
                continue 'again;
            }

            if command(&mut commands, &l.string, b"quit", b"", b"quit the REPL") {
                break 'again;
            }

            if command(
                &mut commands,
                &l.string,
                b"help",
                b"",
                b"print this help message",
            ) {
                print_available_commands(&commands);
                continue 'again;
            }

            print_available_commands(&commands);
            println!("ERROR: unknown command `{}`", stringify(&l.string));
            continue 'again;
        }

        let a: Token;
        let b: Token;
        let cur = l.cur;

        if !l.next() {
            continue 'again;
        }
        a = l.token;
        if !l.next() {
            continue 'again;
        }
        b = l.token;

        l.cur = cur;

        if a == Token::Name && b == Token::Equals {
            if !l.expect(Token::Name) {
                continue 'again;
            }
            let name = symbol(&l.string);
            if !l.expect(Token::Equals) {
                continue 'again;
            }
            let Some(body) = l.parse_expr() else {
                continue 'again;
            };
            if !l.expect(Token::End) {
                continue 'again;
            }
            create_binding(&mut bindings, name, body);
            continue 'again;
        }

        let Some(mut expr) = l.parse_expr() else {
            continue 'again;
        };
        if !l.expect(Token::End) {
            continue 'again;
        }
        for binding in bindings.iter().rev() {
            expr = replace(binding.name, expr, binding.body.clone());
        }

        loop {
            // TODO: can't currently handle SIGINT
            // See original (https://github.com/tsoding/lamb/blob/main/lamb.c#L1377)

            let Some(expr1) = expr.clone().eval1() else {
                continue 'again;
            };
            if Rc::ptr_eq(&expr, &expr1) {
                break;
            }
            expr = expr1;
        }

        print!("RESULT: ");
        expr.trace();
        println!();
    }
}

// Copyright 2025 Emmanuel Amoah (https://emamoah.com/)
//
// Based on original (https://github.com/tsoding/lamb/)
//
// Copyright 2025 Alexey Kutepov <reximkut@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
